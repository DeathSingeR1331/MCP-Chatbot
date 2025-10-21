import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Any
from enum import Enum

import requests
import urllib.parse
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import base64  # For decoding base64 image data
import subprocess

# Gmail MCP Server integration (no custom imports needed)

# LLM Provider imports with availability checking
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq client not available. Install with: pip install groq")

# Ollama removed - only using Groq and Gemini

try:
    from test_notion import (
        test_comment_on_page,
        test_search_pages,
        test_list_databases,
        test_retrieve_comments,
        test_query_database,
        test_retrieve_page,
        test_get_page_blocks,
    )  # reuse working functions
except ImportError:
    # test_notion module not available, define stub functions
    def test_comment_on_page(*args, **kwargs): pass
    def test_search_pages(*args, **kwargs): pass
    def test_list_databases(*args, **kwargs): pass
    def test_retrieve_comments(*args, **kwargs): pass
    def test_query_database(*args, **kwargs): pass
    def test_retrieve_page(*args, **kwargs): pass
    def test_get_page_blocks(*args, **kwargs): pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMProvider(Enum):
    """Enumeration of available LLM providers."""
    GEMINI = "gemini"
    GROQ = "groq"


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.notion_token = os.getenv("NOTION_TOKEN")
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "44565af3b7bbec06cde35bd8512f44bb")
        self.news_api_key = os.getenv("NEWS_API_KEY", "6fb893accc5ac91dd59dac3c82bdaf52")
        # Spotify credentials for automated playback.  If not provided, Spotify
        # integration will be disabled and attempts to play songs will return failure.
        self.spotify_email = os.getenv("SPOTIFY_EMAIL")
        self.spotify_password = os.getenv("SPOTIFY_PASSWORD")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def available_providers(self) -> List[LLMProvider]:
        """Get list of available LLM providers based on API keys and libraries."""
        providers = []
        if GEMINI_AVAILABLE and self.gemini_api_key:
            providers.append(LLMProvider.GEMINI)
        if GROQ_AVAILABLE and self.groq_api_key:
            providers.append(LLMProvider.GROQ)
        return providers


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        logging.info(f"[{self.name}] Initializing server with command: {self.config['command']}")
        logging.info(f"[{self.name}] Server args: {self.config['args']}")
        logging.info(f"[{self.name}] Server env: {self.config.get('env', {})}")
        command = shutil.which("npx") if self.config['command'] == "npx" else self.config['command']
        if not command:
            raise ValueError(f"Command not found: {self.config['command']}")
        server_params = StdioServerParameters(
            command=command,
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None,
        )
        try:
            logging.info(f"[{self.name}] Creating stdio client...")
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            logging.info(f"[{self.name}] Creating client session...")
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            logging.info(f"[{self.name}] Initializing session...")
            init_result = await self.session.initialize()
            self.capabilities = dict(init_result.capabilities) if hasattr(init_result, 'capabilities') else {}
            logging.info(f"[{self.name}] Server initialized successfully with capabilities: {self.capabilities}")
        except Exception as e:
            logging.error(f"[{self.name}] Error initializing server: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        tools_response = await self.session.list_tools()
        tools = []
        supports_progress = self.capabilities and 'progress' in self.capabilities
        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        logging.info(f"Tool '{tool.name}' will support progress tracking")
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        screenshot_tools = ['browser_screenshot', 'browser_take_screenshot', 'puppeteer_screenshot', 'puppeteer_take_screenshot']
        if tool_name in screenshot_tools:
            if 'fullPage' in arguments and isinstance(arguments['fullPage'], str):
                arguments['fullPage'] = arguments['fullPage'].lower() == 'true'
                logging.info(f"[{self.name}] Converted fullPage string to boolean: {arguments['fullPage']}")
            if 'path' not in arguments:
                os.makedirs('images', exist_ok=True)
                timestamp = int(asyncio.get_event_loop().time())
                arguments['path'] = f"images/screenshot_{tool_name}_{timestamp}.png"
                logging.info(f"[{self.name}] Added default image path for {tool_name}: {arguments['path']}")
            elif not arguments['path'].startswith('images/'):
                arguments['path'] = f"images/{arguments['path']}"
                logging.info(f"[{self.name}] Modified path to save in images folder: {arguments['path']}")

        logging.info(f"[{self.name}] Attempting to execute tool '{tool_name}' with arguments: {arguments}")
        attempt = 0
        while attempt < retries:
            try:
                supports_progress = self.capabilities and 'progress' in self.capabilities
                logging.info(f"[{self.name}] Progress tracking supported: {supports_progress}")
                logging.info(f"[{self.name}] Server capabilities: {self.capabilities}")
                if supports_progress:
                    logging.info(f"[{self.name}] Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(tool_name, arguments)
                else:
                    logging.info(f"[{self.name}] Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)
                logging.info(f"[{self.name}] Tool '{tool_name}' executed successfully with result: {result}")
                if tool_name in screenshot_tools and result and hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'mimeType') and content.mimeType == 'image/png' and hasattr(content, 'data'):
                            image_data = base64.b64decode(content.data)
                            with open(arguments['path'], 'wb') as f:
                                f.write(image_data)
                            logging.info(f"[{self.name}] Screenshot saved to: {arguments['path']}")
                            break
                return result
            except Exception as e:
                attempt += 1
                logging.warning(f"[{self.name}] Error executing tool '{tool_name}': {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"[{self.name}] Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"[{self.name}] Max retries reached for tool '{tool_name}'. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await asyncio.wait_for(self.session.__aexit__(None, None, None), timeout=2.0)
                    except asyncio.TimeoutError:
                        logging.warning(f"Session cleanup timeout for {self.name}, forcing cleanup")
                    except asyncio.CancelledError:
                        logging.info(f"Session cleanup cancelled for {self.name} (normal during shutdown)")
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None
                if self.stdio_context:
                    try:
                        await asyncio.wait_for(self.stdio_context.__aexit__(None, None, None), timeout=1.0)
                    except asyncio.TimeoutError:
                        logging.warning(f"Stdio cleanup timeout for {self.name}, forcing cleanup")
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                if param_name == 'path' and self.name in ['browser_screenshot', 'browser_take_screenshot', 'puppeteer_screenshot', 'puppeteer_take_screenshot']:
                    arg_desc += " (use 'images/filename.png' to save in images folder)"
                args_desc.append(arg_desc)
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class MultiLLMClient:
    """Manages communication with multiple LLM providers with fallback mechanism."""

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.current_provider: Optional[LLMProvider] = None
        self.available_providers = config.available_providers
        self.fallback_mode: bool = False  # User controls fallback mode
        if GEMINI_AVAILABLE and self.config.gemini_api_key and genai:
            try:
                genai.configure(api_key=self.config.gemini_api_key)  # type: ignore
            except Exception as e:
                logging.warning(f"Failed to configure Gemini: {e}")
        if GROQ_AVAILABLE and self.config.groq_api_key:
            self.groq_client = Groq(api_key=self.config.groq_api_key)
        try:
            if LLMProvider.GROQ in self.available_providers:
                self.current_provider = LLMProvider.GROQ
            elif LLMProvider.GEMINI in self.available_providers:
                self.current_provider = LLMProvider.GEMINI
        except Exception:
            pass

    def get_response(self, messages: List[Dict[str, str]], provider: Optional[LLMProvider] = None) -> str:
        target_provider = provider or self.current_provider
        if not target_provider:
            if self.available_providers:
                self.current_provider = self.available_providers[0]
                target_provider = self.current_provider
                logging.info(f"Auto-selected LLM provider: {target_provider.value}")
            else:
                return "ERROR: No LLM providers available. Set GROQ_API_KEY or GEMINI_API_KEY in your environment/.env."
        try:
            response = self._get_response_from_provider(messages, target_provider)
            if response:
                logging.info(f"Response from {target_provider.value}")
                return response
        except Exception as e:
            error_msg = f"Provider {target_provider.value} error: {str(e)}"
            logging.error(error_msg)
            if self.fallback_mode:
                logging.info("Fallback mode enabled, trying other providers...")
                for fallback_provider in self.available_providers:
                    if fallback_provider == target_provider:
                        continue
                    try:
                        response = self._get_response_from_provider(messages, fallback_provider)
                        if response:
                            logging.info(f"Fallback successful with {fallback_provider.value}")
                            return response
                    except Exception as fallback_error:
                        logging.warning(f"Fallback provider {fallback_provider.value} also failed: {fallback_error}")
                        continue
            return f"ERROR: {error_msg}"
        return f"WARNING: Failed to get response from {target_provider.value}."

    def _get_response_from_provider(self, messages: List[Dict[str, str]], provider: LLMProvider) -> str:
        if provider == LLMProvider.GEMINI:
            return self._get_gemini_response(messages)
        elif provider == LLMProvider.GROQ:
            return self._get_groq_response(messages)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_gemini_response(self, messages: List[Dict[str, str]]) -> str:
        if not GEMINI_AVAILABLE or not self.config.gemini_api_key:
            raise Exception("Gemini API not available")
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"{message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        try:
            if not genai:
                raise Exception("Gemini module not available")
            model = genai.GenerativeModel('gemini-2.0-flash')  # type: ignore
            response = model.generate_content(prompt)
            return response.text if response.text else "No response generated"
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")

    def _get_groq_response(self, messages: List[Dict[str, str]]) -> str:
        if not GROQ_AVAILABLE or not self.config.groq_api_key:
            raise Exception("Groq API not available")
        try:
            models = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
            ]
            for model in models:
                try:
                    groq_messages = []
                    for msg in messages:
                        groq_messages.append({"role": msg["role"], "content": msg["content"]})
                    response = self.groq_client.chat.completions.create(
                        messages=groq_messages,
                        model=model,
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=1,
                        stream=False,
                    )
                    return response.choices[0].message.content or "No response generated"
                except Exception as model_error:
                    if model == models[-1]:
                        raise model_error
                    logging.warning(f"Model {model} failed: {model_error}. Trying next...")
                    continue
            return "No response generated"
        except Exception as e:
            raise Exception(f"All Groq models failed: {e}")

    def get_weather_data(self, lat: float, lon: float, exclude: str = "minutely,hourly") -> Dict[str, Any]:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,apparent_temperature,is_day,wind_speed_10m,wind_direction_10m,wind_gusts_10m,rain,showers,precipitation,snowfall,surface_pressure,cloud_cover,weather_code,pressure_msl,relative_humidity_2m',
            'hourly': 'temperature_2m,relative_humidity_2m,rain,showers,precipitation,visibility,weather_code,is_day,sunshine_duration',
            'models': 'best_match',
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Weather API error: {e}")
            return {"error": f"Failed to fetch weather data: {e}"}

    def get_weather(self, city: str) -> Dict[str, Any]:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': self.config.weather_api_key,
            'units': 'metric',
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"OpenWeatherMap API error: {e}")
            return {"error": f"Failed to fetch weather data for {city}: {e}"}

    def get_news(self, query: str, max_results: int = 10, language: str = "en") -> Dict[str, Any]:
        url = "https://gnews.io/api/v4/search"
        params = {
            'q': query,
            'lang': language,
            'max': min(max_results, 100),
            'apikey': self.config.news_api_key,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"GNews API error: {e}")
            return {"error": f"Failed to fetch news data for '{query}': {e}"}

    def list_available_providers(self) -> List[str]:
        return [provider.value for provider in self.available_providers]

    def switch_provider(self, provider_name: str) -> bool:
        try:
            provider = LLMProvider(provider_name)
            if provider in self.available_providers:
                self.current_provider = provider
                logging.info(f"Switched to {provider_name}")
                return True
            else:
                logging.warning(f"Provider {provider_name} not available")
                return False
        except ValueError:
            logging.warning(f"Unknown provider: {provider_name}")
            return False


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: MultiLLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: MultiLLMClient = llm_client
        self.tools_loaded: bool = False
        self.all_tools: List[Tool] = []
        self.user_name: Optional[str] = None
        # Use absolute path to ensure consistent file location
        self.user_data_file: str = os.path.abspath("user_data.json")
        logging.info(f"User data file path: {self.user_data_file}")
        self._load_user_data()

    async def cleanup_servers(self) -> None:
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        if cleanup_tasks:
            try:
                # Use a shorter timeout and handle cancellation gracefully
                results = await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=2.0
                )
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        server_name = self.servers[i].name if i < len(self.servers) else f"server_{i}"
                        if isinstance(result, asyncio.CancelledError):
                            logging.info(f"Note: Normal shutdown message for {server_name}: {result}")
                        else:
                            logging.warning(f"Warning during session cleanup for {server_name}: {result}")
            except asyncio.TimeoutError:
                logging.warning("Server cleanup timed out, forcing shutdown")
            except asyncio.CancelledError:
                logging.info("Server cleanup was cancelled (normal during shutdown)")
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    def _load_user_data(self) -> None:
        """Load user data from persistent storage."""
        try:
            logging.info(f"Attempting to load user data from: {self.user_data_file}")
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_name = data.get('user_name')
                    if self.user_name:
                        logging.info(f"✅ Successfully loaded user name: {self.user_name}")
                    else:
                        logging.info("No user name found in data file")
            else:
                logging.info("User data file does not exist yet")
        except Exception as e:
            logging.error(f"❌ Failed to load user data: {e}")
            self.user_name = None

    def _save_user_data(self) -> None:
        """Save user data to persistent storage."""
        try:
            data = {
                'user_name': self.user_name
            }
            logging.info(f"Attempting to save user data to: {self.user_data_file}")
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"✅ Successfully saved user data: {data}")
            
            # Verify the file was created and can be read
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r', encoding='utf-8') as f:
                    verify_data = json.load(f)
                    logging.info(f"✅ Verified saved data: {verify_data}")
            else:
                logging.error("❌ File was not created successfully")
        except Exception as e:
            logging.error(f"❌ Failed to save user data: {e}")
            logging.error(f"File path: {self.user_data_file}")
            logging.error(f"Current working directory: {os.getcwd()}")

    def _extract_name_from_message(self, message: str) -> Optional[str]:
        """Extract name from user message when they set their name."""
        import re
        
        # More comprehensive patterns for name extraction
        patterns = [
            # "my name is abc", "call me abc", "i am abc", "set my name to abc"
            r"(?:my name is|call me|i am|set my name to|remember my name as|change my name to|update my name to)\s+([a-zA-Z0-9\s\-\.']+)",
            # "name is abc", "call me abc"
            r"(?:name is|call me)\s+([a-zA-Z0-9\s\-\.']+)",
            # "abc" (simple case - just the name)
            r"^([a-zA-Z0-9\s\-\.']+)$",
            # "from now on call me abc"
            r"(?:from now on|please)\s+(?:call me|name me)\s+([a-zA-Z0-9\s\-\.']+)",
            # "i want to be called abc"
            r"(?:i want to be called|i want you to call me)\s+([a-zA-Z0-9\s\-\.']+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                name = match.group(1).strip()
                # Clean up the name but preserve it properly
                if name and len(name) > 1:
                    # Remove common words that might be at the beginning or end
                    words = name.split()
                    filtered_words = []
                    
                    for word in words:
                        # Skip common words but keep the actual name
                        if word.lower() not in ['is', 'am', 'to', 'as', 'the', 'a', 'an', 'from', 'now', 'on', 'please', 'me', 'my', 'name']:
                            filtered_words.append(word)
                    
                    if filtered_words:
                        filtered_name = ' '.join(filtered_words).strip()
                        if filtered_name and len(filtered_name) > 1:
                            return filtered_name
        return None

    def _detect_name_setting(self, message: str) -> bool:
        """Detect if user is trying to set their name."""
        lowered = message.lower()
        
        # First check for query patterns (asking about name, not setting it)
        query_patterns = [
            "what's my name", "what is my name", "whats my name",
            "do you remember my name", "do you know my name",
            "tell me my name", "what name do you have"
        ]
        
        # If it's a query, it's not a name setting request
        if any(pattern in lowered for pattern in query_patterns):
            return False
        
        # Then check for setting patterns
        name_indicators = [
            "my name is", "call me", "i am", "set my name", "remember my name",
            "change my name", "update my name", "name is",
            "from now on call me", "please call me", "i want to be called",
            "i want you to call me", "name me", "call me from now on"
        ]
        return any(indicator in lowered for indicator in name_indicators)

    def _handle_name_conflict(self, new_name: str) -> str:
        """Handle conflicts between user-set names and profile information."""
        if self.user_name and self.user_name != new_name:
            logging.info(f"Name conflict detected: current='{self.user_name}', new='{new_name}'")
            return f"✅ Got it! I've updated your name from '{self.user_name}' to '{new_name}'. I'll remember you as {new_name} from now on! 😊"
        else:
            return f"✅ Got it! I'll remember your name as {new_name}. Nice to meet you, {new_name}! 😊"

    def _detect_name_query(self, message: str) -> bool:
        """Detect if user is asking about their name."""
        lowered = message.lower()
        query_patterns = [
            "what's my name", "what is my name", "whats my name",
            "do you remember my name", "do you know my name",
            "tell me my name", "what name do you have",
            "what do you call me", "how do you know me"
        ]
        return any(pattern in lowered for pattern in query_patterns)

    def _handle_name_query(self) -> str:
        """Handle when user asks about their name."""
        if self.user_name:
            return f"Your name is {self.user_name}! I remember that from our previous conversations. 😊"
        else:
            return "I don't have a name set for you yet. You can tell me your name by saying something like 'My name is John' or 'Call me Sarah'."

    async def chat(self, user_message: str) -> str:
        try:
            lowered = user_message.lower()

            # Check if user is asking about their name
            if self._detect_name_query(user_message):
                return self._handle_name_query()
            
            # Check if user is setting their name
            if self._detect_name_setting(user_message):
                extracted_name = self._extract_name_from_message(user_message)
                if extracted_name:
                    # Always prioritize user-set names over profile information
                    old_name = self.user_name
                    self.user_name = extracted_name
                    self._save_user_data()
                    return self._handle_name_conflict(extracted_name)
                else:
                    return "❌ I couldn't understand what name you'd like me to remember. Please try saying something like 'My name is John' or 'Call me Sarah'."

            # Detect Gmail requests: send email, read emails, search emails, open gmail
            if (("gmail" in lowered and "send" in lowered) or 
                ("send email" in lowered) or 
                ("email" in lowered and "send" in lowered) or
                ("compose" in lowered and "email" in lowered) or
                ("write email" in lowered) or
                ("draft email" in lowered) or
                (lowered.startswith("send ") and " to " in lowered and "@" in lowered)):
                email_data = self._extract_gmail_data(lowered)
                if email_data:
                    return await self._handle_gmail_send_via_mcp(email_data)
                else:
                    # If we can't extract email data but it's clearly a send request, try browser automation
                    return await self._handle_gmail_send_via_browser(lowered)
            
            if ("gmail" in lowered and "read" in lowered) or ("read emails" in lowered) or ("show emails" in lowered) or ("open gmail" in lowered) or ("gmail" in lowered and ("open" in lowered or "show" in lowered or "unread" in lowered)) or ("unread message" in lowered and "gmail" in lowered) or ("gmail" in lowered and "message" in lowered):
                return await self._handle_gmail_read_via_mcp(lowered)
            
            if ("gmail" in lowered and "search" in lowered) or ("search emails" in lowered):
                search_query = self._extract_gmail_search_query(lowered)
                if search_query:
                    return await self._handle_gmail_search_via_mcp(search_query)
            
            # Advanced Gmail commands
            gmail_advanced_result = await self._handle_gmail_advanced_commands(lowered)
            if gmail_advanced_result:
                return gmail_advanced_result


            # Detect YouTube playback requests: either mention YouTube and play, or start with "play "
            if ("youtube" in lowered and "play" in lowered) or lowered.startswith("play "):
                song_query = self._extract_youtube_song_query(lowered)
                if song_query:
                    played = await self._play_youtube_song(song_query)
                    return "Playing on YouTube." if played else "Failed to start playback on YouTube."
            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools]) if self.all_tools else "No tools available."
            
            # Add user name to system message if available
            user_name_context = ""
            if self.user_name:
                user_name_context = f"\n\nIMPORTANT: The user's name is {self.user_name}. Always address them by their name when appropriate and use it in your responses to make them more personal and friendly."
            
            # Construct a system message that guides the language model on how to select and
            # invoke tools.  The instructions below emphasise when to use specific tools and
            # when to avoid using browser automation for information retrieval.  In particular,
            # news and general knowledge queries are explicitly covered to prevent the model
            # from falling back to generic browser evaluations (e.g. reading a page title)
            # when a simpler or more appropriate method exists.
            system_message = f"""You are a helpful assistant with access to these tools: \n\n{tools_description}\n\nIMPORTANT RULES:\n1. For NEWS queries (like "latest news", "headlines", "sports news", etc.), ALWAYS use the `get_news` tool with appropriate query parameters.\n2. For WEATHER queries, use the `get_weather` tool with the city name.\n3. For GMAIL queries (like "send email", "read emails", "search emails", "mark emails as read"), DO NOT use browser automation. The system will handle Gmail operations automatically through MCP Gmail tools.\n4. For general knowledge questions, answer directly using your knowledge.\n5. For web browsing tasks, use browser tools like `browser_navigate`, `browser_click`, etc.\n\nWhen you need to use a tool, respond with ONLY this JSON format (nothing else):\n{{\n    \"tool\": \"tool_name\",\n    \"arguments\": {{\n        \"param\": \"value\"\n    }}\n}}\n\nAfter tool execution, provide a natural, conversational response based on the results.{user_name_context}"""
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            llm_response = self.llm_client.get_response(messages)
            result = await self.process_llm_response(llm_response)
            if result != llm_response:
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "system", "content": result})
                final_response = self.llm_client.get_response(messages)
                return final_response
            else:
                return llm_response
        except Exception as e:
            logging.error(f"Error in chat method: {e}")
            return f"Error processing your request: {str(e)}"

    async def process_llm_response(self, llm_response: str) -> str:
        import json
        import re
        tool_calls = []
        try:
            normalized = llm_response
            normalized = re.sub(r"```[a-zA-Z]*", "", normalized)
            normalized = normalized.replace("```", "")
            normalized = normalized.replace('\n', '')
            json_strings = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', normalized)
            for json_str in json_strings:
                try:
                    tool_call = json.loads(json_str)
                    if "tool" in tool_call and "arguments" in tool_call:
                        tool_calls.append(tool_call)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON block: {json_str}, error: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return llm_response
        if not tool_calls:
            try:
                tool_call = json.loads(llm_response)
                if "tool" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                return llm_response
        results = []
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call["tool"]
            arguments = tool_call["arguments"]
            logging.info(f"Executing tool {i+1}/{len(tool_calls)}: {tool_name}")
            logging.info(f"With arguments: {arguments}")
            tool_executed = False
            if tool_name == "play_youtube_song":
                try:
                    query = arguments.get("query", "")
                    if not query:
                        results.append("Missing 'query' for play_youtube_song")
                    else:
                        ok = await self._play_via_local_playwright(query)
                        results.append("YouTube playback started" if ok else "Failed to start YouTube playback")
                except Exception as e:
                    results.append(f"Error starting YouTube playback: {e}")
                tool_executed = True
            elif tool_name == "get_weather":
                try:
                    city = arguments.get("city", "New York")
                    weather_data = self.llm_client.get_weather(city)
                    if "error" in weather_data:
                        result = f"Weather API error: {weather_data['error']}"
                    else:
                        temp = weather_data.get("main", {}).get("temp", "N/A")
                        description = weather_data.get("weather", [{}])[0].get("description", "N/A")
                        humidity = weather_data.get("main", {}).get("humidity", "N/A")
                        result = f"Weather in {city}: {description}, Temperature: {temp}C, Humidity: {humidity}%"
                    results.append(f"Weather tool executed successfully: {result}")
                    tool_executed = True
                except Exception as e:
                    results.append(f"Error executing weather tool: {str(e)}")
                    tool_executed = True
            elif tool_name == "get_news":
                try:
                    query = arguments.get("query", "technology")
                    max_results = arguments.get("max_results", 10)
                    language = arguments.get("language", "en")
                    news_data = self.llm_client.get_news(query, max_results, language)
                    if "error" in news_data:
                        result = f"News API error: {news_data['error']}"
                    else:
                        articles = news_data.get("articles", [])
                        total_articles = news_data.get("totalArticles", 0)
                        if articles:
                            result = f"Found {len(articles)} news articles about '{query}' (Total available: {total_articles}):\n"
                            for i, article in enumerate(articles[:5], 1):
                                title = article.get("title", "No title")
                                source = article.get("source", {}).get("name", "Unknown source")
                                published = article.get("publishedAt", "Unknown date")
                                url = article.get("url", "")
                                result += f"{i}. {title}\n   Source: {source}\n   Published: {published}\n   URL: {url}\n\n"
                        else:
                            result = f"No news articles found for '{query}'"
                    results.append(f"News tool executed successfully: {result}")
                    tool_executed = True
                except Exception as e:
                    results.append(f"Error executing news tool: {str(e)}")
                    tool_executed = True
            if not tool_executed:
                for server in self.servers:
                    try:
                        tools = await server.list_tools()
                        if any(tool.name == tool_name for tool in tools):
                            try:
                                result = await server.execute_tool(tool_name, arguments)
                                if isinstance(result, dict) and 'progress' in result:
                                    progress = result['progress']
                                    total = result['total']
                                    logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                                # Format the result better for different tool types
                                if tool_name == "browser_navigate":
                                    results.append(f"Successfully navigated to: {arguments.get('url', 'unknown URL')}")
                                elif tool_name == "browser_take_screenshot":
                                    results.append(f"Screenshot saved successfully: {result}")
                                elif tool_name == "browser_click":
                                    results.append(f"Successfully clicked on element")
                                elif tool_name == "browser_evaluate":
                                    results.append(f"JavaScript executed successfully: {result}")
                                else:
                                    results.append(f"Tool {tool_name} executed successfully: {result}")
                                tool_executed = True
                                break
                            except Exception as e:
                                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                                logging.error(error_msg)
                                results.append(error_msg)
                                tool_executed = True
                                break
                    except Exception as e:
                        logging.warning(f"Error listing tools from server {server.name}: {e}")
                        continue
            if not tool_executed:
                results.append(f"No server found with tool: {tool_name}")
        if results:
            combined_result = "\n".join(results)
            logging.info(f"All tool executions completed. Results: {combined_result}")
            
            # Instead of returning generic messages, return the actual results
            # This allows the LLM to process the tool results and provide a proper response
            return combined_result
        return llm_response

    async def _play_via_local_playwright(self, query: str) -> bool:
        try:
            code = f'''import asyncio\nfrom playwright.async_api import async_playwright\nVIDEO_QUERY={query!r}\nasync def main():\n    async with async_playwright() as p:\n        browser = await p.chromium.launch(headless=False, args=["--autoplay-policy=no-user-gesture-required"])\n        ctx = await browser.new_context(permissions=["microphone","camera"])\n        page = await ctx.new_page()\n        await page.goto("https://www.youtube.com/results?search_query=" + VIDEO_QUERY.replace(" ", "+"))\n        first = page.locator('ytd-video-renderer a#thumbnail').first\n        await first.click()\n        await page.wait_for_selector("video")\n        try:\n            await page.get_by_role("button", name="Play").click(timeout=2000)\n        except Exception: pass\n        await page.wait_for_timeout(5000)\nasyncio.run(main())\n'''
            proc = subprocess.Popen([sys.executable, "-c", code])
            return proc is not None
        except Exception as e:
            logging.error(f"Local Playwright spawn failed: {e}")
            return False

    async def _play_youtube_song(self, query: str) -> bool:
        """Automate YouTube playback using available browser tools.

        Strategy: navigate to results page, click first result, ensure video.play().
        """
        try:
            search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
            for server in self.servers:
                if not server.session:
                    continue
                try:
                    tools = await server.list_tools()
                    tool_names = {t.name for t in tools}
                except Exception:
                    continue
                # Prefer Playwright, fall back to Puppeteer
                if {"browser_navigate", "browser_evaluate"}.issubset(tool_names):
                    await server.execute_tool("browser_navigate", {"url": search_url})
                    click_js = """
                    () => {
                      const link = document.querySelector('ytd-video-renderer a#video-title, ytd-video-renderer a#thumbnail');
                      if (link && link.href) { window.location.href = link.href; return true; }
                      if (link) { link.click(); return true; }
                      return false;
                    }
                    """
                    # Always click via evaluate for reliability
                    await server.execute_tool("browser_evaluate", {"function": click_js})
                    # Wait a moment for navigation to the video page
                    await asyncio.sleep(1.5)
                    import asyncio as _asyncio
                    for _ in range(12):
                        play_js = """
                        () => {
                          const btn = document.querySelector('.ytp-play-button');
                          if (btn) { btn.click(); }
                          const v = document.querySelector('video');
                          if (!v) return false;
                          try {
                            v.muted = false;
                            const p = v.play?.();
                            if (p && typeof p.then === 'function') { /* ignore */ }
                          } catch (e) {}
                          return !v.paused;
                        }
                        """
                        result = await server.execute_tool("browser_evaluate", {"function": play_js})
                        if result:
                            return True
                        await _asyncio.sleep(0.75)
                    return False
                if {"puppeteer_navigate", "puppeteer_evaluate"}.issubset(tool_names):
                    await server.execute_tool("puppeteer_navigate", {"url": search_url})
                    click_js = """
                    () => {
                      const link = document.querySelector('ytd-video-renderer a#video-title, ytd-video-renderer a#thumbnail');
                      if (link && link.href) { window.location.href = link.href; return true; }
                      if (link) { link.click(); return true; }
                      return false;
                    }
                    """
                    # Always click via evaluate for reliability
                    await server.execute_tool("puppeteer_evaluate", {"script": click_js})
                    # Wait a moment for navigation to the video page
                    await asyncio.sleep(1.5)
                    import asyncio as _asyncio
                    for _ in range(12):
                        play_js = """
                        () => {
                          const btn = document.querySelector('.ytp-play-button');
                          if (btn) { btn.click(); }
                          const v = document.querySelector('video');
                          if (!v) return false;
                          try {
                            v.muted = false;
                            const p = v.play?.();
                            if (p && typeof p.then === 'function') { /* ignore */ }
                          } catch (e) {}
                          return !v.paused;
                        }
                        """
                        result = await server.execute_tool("puppeteer_evaluate", {"script": play_js})
                        if result:
                            return True
                        await _asyncio.sleep(0.75)
                    return False
            return False
        except Exception as e:
            logging.error(f"YouTube playback automation failed: {e}")
        return False

    def _extract_youtube_song_query(self, lowered: str) -> Optional[str]:
        import re
        m = re.search(r"play\s+(.*?)\s+(?:on|in)\s+youtube\b", lowered)
        if m and m.group(1).strip():
            return m.group(1).strip()
        m2 = re.search(r"open\s+youtube.*?play\s+(.+)$", lowered)
        if m2 and m2.group(1).strip():
            return m2.group(1).strip()
        m3 = re.search(r"^play\s+(.+)$", lowered)
        if m3 and m3.group(1).strip():
            return m3.group(1).strip()
        return None

    def _extract_spotify_song_query(self, lowered: str) -> Optional[str]:
        """Extract the song query for Spotify playback.

        This helper looks for patterns like "play <song> on spotify" or
        "play <song> from spotify" in the user's lowercased message.

        Args:
            lowered: The user message in lower case.

        Returns:
            The extracted song name, or None if no suitable pattern is found.
        """
        import re
        # Play <song> on spotify
        m = re.search(r"play\s+(.+?)\s+on\s+spotify", lowered)
        if m and m.group(1).strip():
            return m.group(1).strip()
        # Play <song> from spotify
        m2 = re.search(r"play\s+(.+?)\s+from\s+spotify", lowered)
        if m2 and m2.group(1).strip():
            return m2.group(1).strip()
        # Fallback: if starts with play and mentions spotify anywhere
        if lowered.startswith("play ") and "spotify" in lowered:
            # Remove "play " and trailing "on spotify" or "from spotify" phrases
            q = lowered
            q = re.sub(r"^play\s+", "", q)
            q = re.sub(r"\s+on\s+spotify.*", "", q)
            q = re.sub(r"\s+from\s+spotify.*", "", q)
            return q.strip()
        return None

    async def _play_spotify_song(self, query: str) -> bool:
        """Automate Spotify playback via Playwright.

        This method attempts to play the given song on Spotify using the helper
        defined in the spotify_play module.  Credentials are sourced from
        environment variables or the configuration, if available.  It returns
        True on success and False on failure.

        Args:
            query: The song name to search for and play (e.g. "Romulo Romulo").

        Returns:
            bool: True if playback starts successfully, False otherwise.
        """
        # Lazy import to avoid requiring Playwright if Spotify is never used
        try:
            import spotify_play_final_solution
            play_spotify_song = spotify_play_final_solution.play_spotify_song_final_solution
        except Exception as e:
            logging.error(f"Spotify playback helper could not be imported: {e}")
            return False
        # Gather credentials from configuration or environment
        email = os.getenv("SPOTIFY_EMAIL") or None
        password = os.getenv("SPOTIFY_PASSWORD") or None
        # Prefer credentials from configuration if present
        try:
            cfg_email = getattr(self.llm_client.config, "spotify_email", None)
            cfg_password = getattr(self.llm_client.config, "spotify_password", None)
            if cfg_email:
                email = cfg_email
            if cfg_password:
                password = cfg_password
        except Exception:
            pass
        # If credentials are missing, we cannot automate login/playback
        if not email or not password:
            logging.warning("Spotify credentials not provided; cannot play song on Spotify")
            return False
        try:
            return await play_spotify_song(query, email, password)
        except Exception as e:
            logging.error(f"Error during Spotify playback: {e}")
            return False

    # Removed duplicate _play_youtube_song definition to avoid infinite recursion

    def _extract_gmail_data(self, lowered: str) -> Optional[Dict]:
        """Extract email data from user message for Gmail automation.
        
        This helper looks for patterns like "send email to john@example.com about meeting"
        or "send email to john@example.com subject: meeting body: hello" in the user's message.
        
        Args:
            lowered: The user message in lower case.
            
        Returns:
            Dict containing email data (to, subject, body) or None if no pattern found.
        """
        import re
        
        # Pattern: "send email to john@example.com about meeting"
        pattern = r"send email to ([^\s]+) (?:about|saying|with message) (.+)"
        match = re.search(pattern, lowered)
        if match:
            return {
                "to": match.group(1),
                "subject": "Email from Synapse",
                "body": match.group(2)
            }
        
        # Pattern: "send email to john@example.com subject: meeting body: hello"
        pattern2 = r"send email to ([^\s]+) subject: ([^:]+) body: (.+)"
        match2 = re.search(pattern2, lowered)
        if match2:
            return {
                "to": match2.group(1),
                "subject": match2.group(2),
                "body": match2.group(3)
            }
        
        # Pattern: "send email to john@example.com saying hello world"
        pattern3 = r"send email to ([^\s]+) saying (.+)"
        match3 = re.search(pattern3, lowered)
        if match3:
            return {
                "to": match3.group(1),
                "subject": "Email from Synapse",
                "body": match3.group(2)
            }
        
        # Pattern: "email john@example.com about meeting"
        pattern4 = r"email ([^\s]+) (?:about|saying|with message) (.+)"
        match4 = re.search(pattern4, lowered)
        if match4:
            return {
                "to": match4.group(1),
                "subject": "Email from Synapse",
                "body": match4.group(2)
            }
        
        # Pattern: "compose email to john@example.com about meeting"
        pattern5 = r"compose email to ([^\s]+) (?:about|saying|with message) (.+)"
        match5 = re.search(pattern5, lowered)
        if match5:
            return {
                "to": match5.group(1),
                "subject": "Email from Synapse",
                "body": match5.group(2)
            }
        
        # Pattern: "write email to john@example.com about meeting"
        pattern6 = r"write email to ([^\s]+) (?:about|saying|with message) (.+)"
        match6 = re.search(pattern6, lowered)
        if match6:
            return {
                "to": match6.group(1),
                "subject": "Email from Synapse",
                "body": match6.group(2)
            }
        
        # Pattern: "draft email to john@example.com about meeting"
        pattern7 = r"draft email to ([^\s]+) (?:about|saying|with message) (.+)"
        match7 = re.search(pattern7, lowered)
        if match7:
            return {
                "to": match7.group(1),
                "subject": "Email from Synapse",
                "body": match7.group(2)
            }
        
        # Pattern: "send hello to john@example.com" or "send message to john@example.com"
        pattern8 = r"send ([^\s]+) to ([^\s]+)"
        match8 = re.search(pattern8, lowered)
        if match8:
            return {
                "to": match8.group(2),
                "subject": "Email from Synapse",
                "body": match8.group(1)
            }
        
        # Pattern: "send hello world to john@example.com" (multi-word message)
        pattern9 = r"send (.+?) to ([^\s]+)"
        match9 = re.search(pattern9, lowered)
        if match9:
            return {
                "to": match9.group(2),
                "subject": "Email from Synapse",
                "body": match9.group(1)
            }
        
        return None

    def _extract_gmail_search_query(self, lowered: str) -> Optional[str]:
        """Extract search query from user message for Gmail search.
        
        This helper looks for patterns like "search emails for meeting" or
        "search emails about project" in the user's message.
        
        Args:
            lowered: The user message in lower case.
            
        Returns:
            The extracted search query or None if no pattern found.
        """
        import re
        
        # Pattern: "search emails for meeting"
        pattern = r"search emails (?:for|about) (.+)"
        match = re.search(pattern, lowered)
        if match:
            return match.group(1)
        
        # Pattern: "search gmail for meeting"
        pattern2 = r"search gmail (?:for|about) (.+)"
        match2 = re.search(pattern2, lowered)
        if match2:
            return match2.group(1)
        
        return None



    async def _handle_gmail_send_via_mcp(self, email_data: Dict) -> str:
        """Handle Gmail send via MCP AutoAuth server."""
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            # Check for attachments
            attachment_paths = self._extract_attachment_paths(email_data.get('body', ''))
            
            # Prepare email data for MCP tool
            email_args = {
                "to": [email_data['to']],
                "subject": email_data['subject'],
                "body": email_data['body'],
                "mimeType": "text/plain"
            }
            
            # Add attachments if found
            if attachment_paths:
                email_args["attachments"] = attachment_paths
            
            # Execute send_email tool
            result = await gmail_server.execute_tool("send_email", email_args)
            
            if result and result.content:
                # Extract text content from MCP response
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                
                if "successfully" in text_content.lower():
                    return f"📧 Email sent successfully! {text_content}"
                else:
                    return f"❌ Failed to send email: {text_content}"
            else:
                return "❌ Failed to send email: No response from Gmail server"
                
        except Exception as e:
            logging.error(f"Gmail MCP send failed: {e}")
            return f"❌ Gmail send failed: {str(e)}"

    async def _handle_gmail_read_via_mcp(self, lowered: str) -> str:
        """Handle Gmail read via MCP AutoAuth server."""
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            # Determine search query based on user request
            if "unread" in lowered:
                search_query = "is:unread"
            elif "first unread" in lowered:
                search_query = "is:unread"
            else:
                search_query = "in:inbox"
            
            # Use search_emails to get emails
            search_args = {
                "query": search_query,
                "maxResults": 10
            }
            
            result = await gmail_server.execute_tool("search_emails", search_args)
            
            if result and result.content:
                # Extract text content from MCP response
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                
                # Parse the email data from the text response
                emails = self._parse_gmail_search_response(text_content)
                
                # Handle first unread message request
                if "first unread" in lowered and emails:
                    first_email = emails[0]
                    sender = first_email.get("from", "Unknown")
                    subject = first_email.get("subject", "No Subject")
                    preview = first_email.get("snippet", "")
                    
                    response = f"📧 First Unread Message:\n\n"
                    response += f"From: {sender}\n"
                    response += f"Subject: {subject}\n"
                    if preview:
                        response += f"Preview: {preview}"
                    
                    return response
                
                # Format email list
                email_list = []
                for i, email in enumerate(emails[:5], 1):
                    sender = email.get("from", "Unknown")
                    subject = email.get("subject", "No Subject")
                    date = email.get("date", "")
                    
                    email_text = f"{i}. From: {sender}\n   Subject: {subject}\n   Date: {date}"
                    email_list.append(email_text)
                
                email_display = "\n\n".join(email_list)
                total_count = len(emails)
                
                response = f"📧 Found {total_count} emails"
                if "unread" in lowered:
                    response += " (unread)"
                response += f":\n\n{email_display}"
                
                return response
            else:
                if "unread" in lowered:
                    return "📧 No unread messages found in your Gmail inbox."
                else:
                    return "📧 No emails found in your Gmail inbox."
                
        except Exception as e:
            logging.error(f"Gmail MCP read failed: {e}")
            return f"❌ Gmail read failed: {str(e)}"

    async def _handle_gmail_search_via_mcp(self, query: str) -> str:
        """Handle Gmail search via MCP AutoAuth server."""
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            search_args = {
                "query": query,
                "maxResults": 10
            }
            
            result = await gmail_server.execute_tool("search_emails", search_args)
            
            if result and result.content:
                # Extract text content from MCP response
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                
                # Parse the email data from the text response
                emails = self._parse_gmail_search_response(text_content)
                
                if emails:
                    email_list = []
                    for i, email in enumerate(emails, 1):
                        sender = email.get("from", "Unknown")
                        subject = email.get("subject", "No Subject")
                        date = email.get("date", "")
                        
                        email_text = f"{i}. From: {sender}\n   Subject: {subject}\n   Date: {date}"
                        email_list.append(email_text)
                    
                    email_display = "\n\n".join(email_list)
                    return f"📧 Search results for '{query}':\n\n{email_display}"
                else:
                    return f"📧 No emails found matching '{query}'"
            else:
                return f"📧 No emails found matching '{query}'"
                
        except Exception as e:
            logging.error(f"Gmail MCP search failed: {e}")
            return f"❌ Gmail search failed: {str(e)}"

    def _get_gmail_server(self):
        """Get the Gmail MCP server."""
        for server in self.servers:
            if server.name == "gmail":
                return server
        return None


    def _extract_attachment_paths(self, text: str) -> List[str]:
        """Extract file paths from user message."""
        import re
        
        # Look for file paths in the text
        file_patterns = [
            r'[A-Za-z]:\\[^\\s]+',  # Windows paths
            r'/[^\\s]+',            # Unix paths
            r'\./[^\\s]+',          # Relative paths
        ]
        
        attachments = []
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            attachments.extend(matches)
        
        # Filter for actual files (basic check)
        valid_attachments = []
        for path in attachments:
            if os.path.exists(path) and os.path.isfile(path):
                valid_attachments.append(path)
        
        return valid_attachments

    def _parse_gmail_search_response(self, text_content: str) -> List[Dict]:
        """Parse Gmail search response text into email objects."""
        emails = []
        
        # Split by double newlines to get individual emails
        email_blocks = text_content.strip().split('\n\n')
        
        for block in email_blocks:
            if not block.strip():
                    continue
                    
            lines = block.strip().split('\n')
            email_data = {}
            
            for line in lines:
                if line.startswith('ID: '):
                    email_data['id'] = line[4:].strip()
                elif line.startswith('Subject: '):
                    email_data['subject'] = line[9:].strip()
                elif line.startswith('From: '):
                    email_data['from'] = line[6:].strip()
                elif line.startswith('Date: '):
                    email_data['date'] = line[6:].strip()
            
            if email_data:  # Only add if we have some data
                emails.append(email_data)
        
        return emails

    async def _handle_gmail_advanced_commands(self, lowered: str) -> str:
        """Handle advanced Gmail commands via MCP."""
        
        # Draft email
        if "draft" in lowered and "email" in lowered:
            return await self._handle_gmail_draft_via_mcp(lowered)
        
        # List labels
        if "list labels" in lowered or "show labels" in lowered:
            return await self._handle_gmail_labels_via_mcp()
        
        # Create label
        if "create label" in lowered:
            return await self._handle_gmail_create_label_via_mcp(lowered)
        
        # Delete email
        if "delete email" in lowered:
            return await self._handle_gmail_delete_via_mcp(lowered)
        
        # Mark as read/unread
        if "mark as read" in lowered or "mark as unread" in lowered:
            return await self._handle_gmail_mark_via_mcp(lowered)
        
        # List filters
        if "list filters" in lowered or "show filters" in lowered:
            return await self._handle_gmail_list_filters_via_mcp()
        
        # Create filter
        if "create filter" in lowered:
            return await self._handle_gmail_create_filter_via_mcp(lowered)
        
        return None

    async def _handle_gmail_draft_via_mcp(self, lowered: str) -> str:
        """Create email draft via MCP."""
        email_data = self._extract_gmail_data(lowered)
        if not email_data:
            return "❌ Could not extract email data for draft"
        
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            draft_args = {
                "to": [email_data['to']],
                "subject": email_data['subject'],
                "body": email_data['body']
            }
            
            result = await gmail_server.execute_tool("draft_email", draft_args)
            
            if result and result.content:
                # Extract text content from MCP response
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                
                if "successfully" in text_content.lower():
                    return f"📝 Email draft created successfully! {text_content}"
                else:
                    return f"❌ Failed to create draft: {text_content}"
            else:
                return "❌ Failed to create draft: No response from Gmail server"
                
        except Exception as e:
            return f"❌ Draft creation failed: {str(e)}"

    async def _handle_gmail_labels_via_mcp(self) -> str:
        """List Gmail labels via MCP."""
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            result = await gmail_server.execute_tool("list_email_labels", {})
            
            if result and result.content:
                # Extract text content from MCP response
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                
                return f"📋 Gmail Labels:\n\n{text_content}"
            else:
                return "📋 No labels found"
                
        except Exception as e:
            return f"❌ Failed to list labels: {str(e)}"

    async def _handle_gmail_create_label_via_mcp(self, lowered: str) -> str:
        """Create Gmail label via MCP."""
        # Extract label name from user message
        import re
        match = re.search(r'create label[:\s]+([^\s]+)', lowered)
        if not match:
            return "❌ Please specify label name. Example: 'create label Work'"
        
        label_name = match.group(1)
        
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            label_args = {
                "name": label_name,
                "messageListVisibility": "show",
                "labelListVisibility": "labelShow"
            }
            
            result = await gmail_server.execute_tool("create_label", label_args)
            
            if result and result.content:
                # Extract text content from MCP response
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                
                if "successfully" in text_content.lower():
                    return f"🏷️ Label '{label_name}' created successfully! {text_content}"
                else:
                    return f"❌ Failed to create label: {text_content}"
            else:
                return "❌ Failed to create label: No response from Gmail server"
                
        except Exception as e:
            return f"❌ Label creation failed: {str(e)}"

    async def _handle_gmail_delete_via_mcp(self, lowered: str) -> str:
        """Delete Gmail email via MCP."""
        # Extract message ID from user message
        import re
        match = re.search(r'delete email[:\s]+([^\s]+)', lowered)
        if not match:
            return "❌ Please specify message ID. Example: 'delete email 182ab45cd67ef'"
        
        message_id = match.group(1)
        
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            delete_args = {
                "messageId": message_id
            }
            
            result = await gmail_server.execute_tool("delete_email", delete_args)
            
            if result and result.get("success"):
                return f"🗑️ Email {message_id} deleted successfully!"
            else:
                return f"❌ Failed to delete email: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Email deletion failed: {str(e)}"

    async def _handle_gmail_mark_via_mcp(self, lowered: str) -> str:
        """Mark Gmail emails as read/unread via MCP."""
        # Extract message ID from user message
        import re
        match = re.search(r'mark.*?([^\s]+).*?(?:as read|as unread)', lowered)
        if not match:
            return "❌ Please specify message ID. Example: 'mark email 182ab45cd67ef as read'"
        
        message_id = match.group(1)
        is_read = "as read" in lowered
        
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            modify_args = {
                "messageId": message_id,
                "addLabelIds": [] if is_read else ["UNREAD"],
                "removeLabelIds": ["UNREAD"] if is_read else []
            }
            
            result = await gmail_server.execute_tool("modify_email", modify_args)
            
            if result and result.get("success"):
                status = "read" if is_read else "unread"
                return f"✅ Email {message_id} marked as {status}!"
            else:
                return f"❌ Failed to mark email: {result.get('error', 'Unknown error')}"
                    
        except Exception as e:
            return f"❌ Email marking failed: {str(e)}"

    async def _handle_gmail_list_filters_via_mcp(self) -> str:
        """List Gmail filters via MCP."""
        try:
            gmail_server = self._get_gmail_server()
            if not gmail_server:
                return "❌ Gmail MCP server not available"
            
            result = await gmail_server.execute_tool("list_filters", {})
            
            if result and result.get("filters"):
                filters = result["filters"]
                filter_list = []
                
                for filter_item in filters:
                    filter_id = filter_item.get("id", "Unknown")
                    criteria = filter_item.get("criteria", {})
                    action = filter_item.get("action", {})
                    
                    criteria_text = ", ".join([f"{k}: {v}" for k, v in criteria.items()])
                    action_text = ", ".join([f"{k}: {v}" for k, v in action.items()])
                    
                    filter_list.append(f"• ID: {filter_id}\n  Criteria: {criteria_text}\n  Action: {action_text}")
                
                filter_display = "\n\n".join(filter_list)
                return f"🔍 Gmail Filters:\n\n{filter_display}"
            else:
                return "🔍 No filters found"
                
        except Exception as e:
            return f"❌ Failed to list filters: {str(e)}"

    async def _handle_gmail_create_filter_via_mcp(self, lowered: str) -> str:
        """Create Gmail filter via MCP."""
        # This is a simplified version - in practice, you'd want more sophisticated parsing
        return "❌ Filter creation requires specific parameters. Please use the Gmail interface for complex filters."



    async def _handle_gmail_send_via_browser(self, lowered: str) -> str:
        """Handle Gmail send via browser automation as fallback."""
        try:
            # Extract email data from the message
            email_data = self._extract_gmail_data(lowered)
            if not email_data:
                return "❌ Could not extract email data. Please specify recipient, subject, and body clearly."
            
            # Find a browser automation server
            browser_server = None
            for server in self.servers:
                if not server.session:
                    continue
                try:
                    tools = await server.list_tools()
                    tool_names = {t.name for t in tools}
                    if {"browser_navigate", "browser_evaluate", "browser_click", "browser_type"}.issubset(tool_names):
                        browser_server = server
                        break
                except Exception:
                    continue
                    
            if not browser_server:
                return "❌ No browser automation server available for Gmail sending."
            
            # Navigate to Gmail
            await browser_server.execute_tool("browser_navigate", {"url": "https://mail.google.com"})
            await asyncio.sleep(3)
            
            # Click Compose button
            compose_js = """
            () => {
                const composeButton = document.querySelector('[role="button"][aria-label*="Compose"], [role="button"][aria-label*="compose"], .T-I.T-I-KE.L3');
                if (composeButton) {
                    composeButton.click();
                    return true;
                }
                return false;
            }
            """
            await browser_server.execute_tool("browser_evaluate", {"function": compose_js})
            await asyncio.sleep(2)
            
            # Fill recipient field
            recipient_js = f"""
            () => {{
                const selectors = [
                    'input[aria-label*="To"]',
                    'input[placeholder*="To"]', 
                    '.vR input',
                    'input[aria-label="To recipients"]',
                    'input[aria-label="To"]',
                    'div[aria-label*="To"] input',
                    'div[role="combobox"] input',
                    'input[type="text"]'
                ];
                
                let toField = null;
                for (const selector of selectors) {{
                    toField = document.querySelector(selector);
                    if (toField && toField.offsetParent !== null) {{
                        break;
                    }}
                }}
                
                if (toField) {{
                    toField.focus();
                    toField.click();
                    toField.value = "{email_data['to']}";
                    toField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    toField.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    toField.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                    toField.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', bubbles: true }}));
                    return true;
                }}
                return false;
            }}
            """
            recipient_result = await browser_server.execute_tool("browser_evaluate", {"function": recipient_js})
            logging.info(f"Recipient field fill result: {recipient_result}")
            await asyncio.sleep(2)
            
            # Fill subject field
            subject_js = f"""
            () => {{
                const subjectField = document.querySelector('input[aria-label*="Subject"], input[placeholder*="Subject"], .aoT input');
                if (subjectField) {{
                    subjectField.focus();
                    subjectField.value = "{email_data['subject']}";
                    subjectField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    return true;
                }}
                return false;
            }}
            """
            await browser_server.execute_tool("browser_evaluate", {"function": subject_js})
            await asyncio.sleep(1)
            
            # Fill body field
            body_js = f"""
            () => {{
                const bodyField = document.querySelector('div[aria-label*="Message Body"], div[contenteditable="true"], .Am.Al.editable');
                if (bodyField) {{
                    bodyField.focus();
                    bodyField.innerHTML = "{email_data['body'].replace('"', '\\"')}";
                    bodyField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    return true;
                }}
                return false;
            }}
            """
            await browser_server.execute_tool("browser_evaluate", {"function": body_js})
            await asyncio.sleep(2)
            
            # Click Send button
            send_js = """
            () => {
                const selectors = [
                    'button[aria-label*="Send"]',
                    'button[aria-label*="send"]',
                    'div[role="button"][aria-label*="Send"]',
                    'div[role="button"][aria-label*="send"]',
                    '.T-I.J-J5-Ji.aoO.T-I-atl.L3'
                ];
                
                let sendButton = null;
                for (const selector of selectors) {
                    sendButton = document.querySelector(selector);
                    if (sendButton && sendButton.offsetParent !== null) {
                        break;
                    }
                }
                
                if (!sendButton) {
                    const buttons = document.querySelectorAll('button, div[role="button"]');
                    for (const button of buttons) {
                        if (button.textContent && button.textContent.toLowerCase().includes('send')) {
                            sendButton = button;
                            break;
                        }
                    }
                }
                
                if (sendButton) {
                    sendButton.click();
                    return true;
                }
                return false;
            }
            """
            send_result = await browser_server.execute_tool("browser_evaluate", {"function": send_js})
            logging.info(f"Send button click result: {send_result}")
            await asyncio.sleep(3)
            
            return f"📧 Email sent successfully via browser automation to {email_data['to']}!"
                    
        except Exception as e:
            logging.error(f"Gmail browser automation failed: {e}")
            return f"❌ Gmail send failed via browser automation: {str(e)}"


    async def _search_gmail_via_browser_old(self, query: str) -> List[Dict]:
        """Search emails via Gmail.com browser automation - SAME PATTERN AS YOUTUBE.
        
        This method automates Gmail email searching using the same browser automation
        pattern as YouTube. It navigates to Gmail, performs search, and extracts results.
        
        Args:
            query: Search query string.
            
        Returns:
            List of email dictionaries matching the search query.
        """
        try:
            for server in self.servers:
                if not server.session:
                    continue
                try:
                    tools = await server.list_tools()
                    tool_names = {t.name for t in tools}
                except Exception:
                    continue
                    
                if {"browser_navigate", "browser_evaluate"}.issubset(tool_names):
                    # Navigate to Gmail
                    await server.execute_tool("browser_navigate", {"url": "https://mail.google.com"})
                    await asyncio.sleep(3)
                    
                    # Click search box
                    search_js = """
                    () => {
                        const searchBox = document.querySelector('input[aria-label="Search mail"]');
                        if (searchBox) {
                            searchBox.click();
                            searchBox.focus();
                            return true;
                        }
                        return false;
                    }
                    """
                    await server.execute_tool("browser_evaluate", {"function": search_js})
                    await asyncio.sleep(1)
                    
                    # Type search query
                    type_search_js = f"""
                    () => {{
                        const searchBox = document.querySelector('input[aria-label="Search mail"]');
                        if (searchBox) {{
                            searchBox.value = "{query}";
                            searchBox.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            searchBox.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter' }}));
                            return true;
                        }}
                        return false;
                    }}
                    """
                    await server.execute_tool("browser_evaluate", {"function": type_search_js})
                    await asyncio.sleep(3)
                    
                    # Extract search results using the same improved extraction
                    read_emails_js = """
                    () => {
                        const emails = [];
                        
                        // Try multiple selectors for Gmail email rows
                        const selectors = [
                            'tr[role="row"]',
                            'div[role="main"] tr',
                            '.zA',
                            '[data-legacy-thread-id]',
                            'tr[jscontroller]'
                        ];
                        
                        let emailElements = [];
                        for (const selector of selectors) {
                            emailElements = document.querySelectorAll(selector);
                            if (emailElements.length > 0) break;
                        }
                        
                        for (let i = 0; i < Math.min(emailElements.length, 10); i++) {
                            const email = emailElements[i];
                            
                            // Try multiple selectors for sender
                            let sender = 'Unknown';
                            const senderSelectors = [
                                'span[email]',
                                '.yW span[email]',
                                '.yW span',
                                '[data-hovercard-id]',
                                'span[title]'
                            ];
                            
                            for (const sel of senderSelectors) {
                                const senderEl = email.querySelector(sel);
                                if (senderEl) {
                                    sender = senderEl.getAttribute('email') || senderEl.textContent || senderEl.getAttribute('title') || 'Unknown';
                                    if (sender && sender !== 'Unknown') break;
                                }
                            }
                            
                            // Try multiple selectors for subject
                            let subject = 'No Subject';
                            const subjectSelectors = [
                                '.y6 span[title]',
                                '.y6 span',
                                'span[title]',
                                '.bog',
                                '[data-legacy-thread-id] span'
                            ];
                            
                            for (const sel of subjectSelectors) {
                                const subjectEl = email.querySelector(sel);
                                if (subjectEl) {
                                    subject = subjectEl.textContent || subjectEl.getAttribute('title') || 'No Subject';
                                    if (subject && subject !== 'No Subject') break;
                                }
                            }
                            
                            // Try multiple selectors for preview
                            let preview = '';
                            const previewSelectors = [
                                '.y2',
                                '.yP',
                                '.y2 span',
                                '.yP span'
                            ];
                            
                            for (const sel of previewSelectors) {
                                const previewEl = email.querySelector(sel);
                                if (previewEl) {
                                    preview = previewEl.textContent || '';
                                    if (preview) break;
                                }
                            }
                            
                            emails.push({
                                sender: sender.trim(),
                                subject: subject.trim(),
                                preview: preview.trim()
                            });
                        }
                        
                        return emails;
                    }
                    """
                    
                    result = await server.execute_tool("browser_evaluate", {"function": read_emails_js})
                    return result if result else []
                    
        except Exception as e:
            logging.error(f"Gmail search automation failed: {e}")
            return []

    async def _mark_gmail_read_via_browser(self) -> bool:
        """Mark emails as read via Gmail.com browser automation - SAME PATTERN AS YOUTUBE.
        
        This method automates marking Gmail emails as read using the same browser
        automation pattern as YouTube. It navigates to Gmail and marks all visible
        emails as read.
        
        Returns:
            True if emails were marked as read successfully, False otherwise.
        """
        try:
            for server in self.servers:
                if not server.session:
                    continue
                try:
                    tools = await server.list_tools()
                    tool_names = {t.name for t in tools}
                except Exception:
                    continue
                    
                if {"browser_navigate", "browser_evaluate"}.issubset(tool_names):
                    # Navigate to Gmail
                    await server.execute_tool("browser_navigate", {"url": "https://mail.google.com"})
                    await asyncio.sleep(3)
                    
                    # Select all emails
                    select_all_js = """
                    () => {
                        const selectAllBtn = document.querySelector('input[type="checkbox"][aria-label="Select all"]');
                        if (selectAllBtn) {
                            selectAllBtn.click();
                            return true;
                        }
                        return false;
                    }
                    """
                    await server.execute_tool("browser_evaluate", {"function": select_all_js})
                    await asyncio.sleep(1)
                    
                    # Mark as read
                    mark_read_js = """
                    () => {
                        const markReadBtn = document.querySelector('[aria-label*="Mark as read"]');
                        if (markReadBtn) {
                            markReadBtn.click();
                            return true;
                        }
                        return false;
                    }
                    """
                    result = await server.execute_tool("browser_evaluate", {"function": mark_read_js})
                    await asyncio.sleep(2)
                    
                    return result
                    
        except Exception as e:
            logging.error(f"Gmail mark as read automation failed: {e}")
            return False

    async def start(self) -> None:
        try:
            print("\n" + "="*70)
            print("Starting Multi-LLM Chat Assistant...")
            print("="*70)
            print("\nInitializing MCP servers...")
            initialized_servers = 0
            for server in self.servers:
                try:
                    await server.initialize()
                    initialized_servers += 1
                    print(f"  + Server '{server.name}' initialized")
                except Exception as e:
                    logging.error(f"  - Failed to initialize server '{server.name}': {e}")
            if initialized_servers == 0:
                print("\nWARNING: No MCP servers were initialized. Continuing without tools...")
            print("\nLoading tools...")
            self.all_tools = []
            for server in self.servers:
                if server.session:
                    try:
                        tools = await server.list_tools()
                        self.all_tools.extend(tools)
                        print(f"  + Loaded {len(tools)} tools from '{server.name}'")
                    except Exception as e:
                        logging.warning(f"  - Could not load tools from '{server.name}': {e}")
            self.tools_loaded = len(self.all_tools) > 0
            print(f"\nTotal tools available: {len(self.all_tools)}")
            print(f"\nAvailable tools from each server:")
            for server in self.servers:
                if server.session:
                    try:
                        tools = await server.list_tools()
                        tool_names = [tool.name for tool in tools]
                        print(f"  {server.name}: {tool_names}")
                    except Exception as e:
                        print(f"  ERROR {server.name}: Error listing tools - {e}")
            print("\nSELECT YOUR LLM PROVIDER:")
            print("="*70)
            available_providers = self.llm_client.list_available_providers()
            if not available_providers:
                print("ERROR: No LLM providers available! Please check your API keys and installations.")
                return
            for idx, provider in enumerate(available_providers, 1):
                provider_display = {
                    "gemini": "Gemini 2.0 Flash Experimental (Google API)",
                    "groq": "Groq - Llama 3.3 70B (Fast API)",
                    "ollama_mistral": "Ollama - Mistral Latest (Local)",
                    "ollama_qwen": "Ollama - Qwen 2.5 Latest (Local)",
                }
                print(f"  [{idx}] {provider_display.get(provider, provider)}")
            print("="*70)
            while True:
                try:
                    choice = input(f"\nEnter your choice (1-{len(available_providers)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_providers):
                        selected_provider = available_providers[choice_idx]
                        if self.llm_client.switch_provider(selected_provider):
                            print(f"\nSelected: {selected_provider}")
                            current_provider = self.llm_client.current_provider
                            print(f"Current provider: {current_provider.value if current_provider else 'None'}")
                            break
                        else:
                            print("Failed to switch provider. Please try again.")
                    else:
                        print(f"Invalid choice. Please enter a number between 1 and {len(available_providers)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    return
            print("\nType /help for commands or start chatting!")
            print("="*70)
            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools]) if self.all_tools else "No tools available."
            
            # Add user name to system message if available
            user_name_context = ""
            if self.user_name:
                user_name_context = f"\n\nIMPORTANT: The user's name is {self.user_name}. Always address them by their name when appropriate and use it in your responses to make them more personal and friendly."
            
            # Construct a concise system message for the interactive CLI.  This message
            # instructs the model how to choose tools and includes specific guidance for
            # handling news and general knowledge queries without resorting to browser
            # automation.  It retains the instructions on formatting tool calls and
            # handling screenshot tools.
            system_message = (
                f"You are a helpful assistant with access to these tools: \n\n{tools_description}\n\n"
                "NEWS QUERIES:\n"
                "- If the user asks for news, headlines, latest updates, sports news, or anything similar, you MUST use the `get_news` tool. DO NOT use browser tools (like `browser_evaluate` or `browser_navigate`) to scrape web pages for news or headlines.\n\n"
                "GENERAL QUERIES:\n"
                "- If the user asks a general question (e.g., `Who is the captain of the new ODI World Cup?`, `What is the capital of France?`, definitions, biographies, etc.), answer directly using your knowledge and reasoning. Do not use browser tools to scrape web pages for such questions.\n\n"
                "Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, respond with the exact JSON object format below. For multiple tools or multi-step tasks, provide multiple JSON objects, each on a separate line. Nothing else in the response:\n\n"
                "{\n"
                "    \"tool\": \"tool-name\",\n"
                "    \"arguments\": {\n"
                "        \"argument-name\": \"value\"\n"
                "    }\n"
                "}\n\n"
                "For screenshot tools (browser_take_screenshot, puppeteer_screenshot, etc.), always use a path starting with 'images/' (e.g., 'images/screenshot.png') to save in the images folder. Use boolean values for fullPage parameter (true/false, not \"true\"/\"false\").\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above.{user_name_context}"
            )
            messages = [
                {"role": "system", "content": system_message},
            ]
            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                    if not user_input:
                        continue
                    if user_input.startswith('/'):
                        should_exit = await self._handle_command(user_input)
                        if should_exit:
                            break
                        continue
                    
                    # Check if user is asking about their name
                    if self._detect_name_query(user_input):
                        print(self._handle_name_query())
                        continue
                    
                    # Check if user is setting their name
                    if self._detect_name_setting(user_input):
                        extracted_name = self._extract_name_from_message(user_input)
                        if extracted_name:
                            # Always prioritize user-set names over profile information
                            self.user_name = extracted_name
                            self._save_user_data()
                            print(self._handle_name_conflict(extracted_name))
                            continue
                        else:
                            print("❌ I couldn't understand what name you'd like me to remember. Please try saying something like 'My name is John' or 'Call me Sarah'.")
                            continue
                    
                    messages.append({"role": "user", "content": user_input})
                    llm_response = self.llm_client.get_response(messages)
                    assistant_output = llm_response
                    try:
                        data = json.loads(assistant_output)
                        if "tool" in data:
                            tool_name = data["tool"]
                            args = data.get("arguments", {})
                            if tool_name == "API-create-a-comment":
                                page_id = args["parent"]["page_id"]
                                comment_text = args["rich_text"][0]["text"]["content"]
                                result = test_comment_on_page(page_id, comment_text)
                                if result:
                                    print("✅ Comment added successfully to Notion!")
                            elif tool_name == "API-retrieve-a-comment":
                                page_id = args.get("block_id", "")
                                if page_id:
                                    comments = test_retrieve_comments(page_id)
                                    if comments:
                                        print(f"✅ Retrieved {len(comments)} comments from Notion!")
                            elif tool_name == "API-post-search":
                                query = args.get("query", "")
                                filter_type = args.get("filter", {}).get("value", "page")
                                if filter_type == "page":
                                    pages = test_search_pages()
                                    if pages:
                                        print(f"✅ Found {len(pages)} pages in Notion!")
                                elif filter_type == "database":
                                    databases = test_list_databases()
                                    if databases:
                                        print(f"✅ Found {len(databases)} databases in Notion!")
                            elif tool_name == "API-post-database-query":
                                database_id = args.get("database_id", "")
                                if database_id:
                                    entries = test_query_database(database_id)
                                    if entries:
                                        print(f"✅ Queried database and found {len(entries)} entries!")
                            elif tool_name == "API-retrieve-a-page":
                                page_id = args.get("page_id", "")
                                if page_id:
                                    page_data = test_retrieve_page(page_id)
                                    if page_data:
                                        print("✅ Retrieved page details from Notion!")
                            elif tool_name == "API-get-block-children":
                                block_id = args.get("block_id", "")
                                if block_id:
                                    blocks = test_get_page_blocks(block_id)
                                    if blocks:
                                        print(f"✅ Retrieved {len(blocks)} content blocks from Notion!")
                            else:
                                print("💬 Assistant:", assistant_output)
                        else:
                            print("💬 Assistant:", assistant_output)
                    except json.JSONDecodeError:
                        print("💬 Assistant:", assistant_output)
                    result = await self.process_llm_response(llm_response)
                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})
                        final_response = self.llm_client.get_response(messages)
                        print(f"\n🔧 Final Response: {final_response}")
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        messages.append({"role": "assistant", "content": llm_response})
                except KeyboardInterrupt:
                    print("\n\n👋 Exiting...")
                    break
                except Exception as e:
                    logging.error(f"Error in chat loop: {e}")
                    print(f"\n❌ An error occurred: {e}\n")
        finally:
            print("\n🧹 Cleaning up...")
            await self.cleanup_servers()
            print("✓ Cleanup complete")

    async def _handle_command(self, command: str) -> bool:
        parts = command.split()
        cmd = parts[0].lower()
        if cmd in ['/help', '/h']:
            self.show_help()
        elif cmd in ['/providers', '/p']:
            print("\n🔧 Available LLM Providers:")
            for provider in self.llm_client.list_available_providers():
                current = "← CURRENT" if self.llm_client.current_provider and provider == self.llm_client.current_provider.value else ""
                print(f"  • {provider} {current}")
            print()
        elif cmd in ['/switch'] and len(parts) > 1:
            provider_name = parts[1]
            if self.llm_client.switch_provider(provider_name):
                print(f"✅ Switched to {provider_name}\n")
            else:
                print(f"❌ Provider '{provider_name}' not available\n")
        elif cmd in ['/current', '/c']:
            current = self.llm_client.current_provider
            if current:
                print(f"🎯 Current provider: {current.value}\n")
            else:
                print("🎯 Using automatic fallback (no specific provider set)\n")
        elif cmd in ['/name', '/n']:
            if self.user_name:
                print(f"👤 Your name: {self.user_name}")
                print(f"📁 Stored in: {self.user_data_file}")
                print(f"📄 File exists: {os.path.exists(self.user_data_file)}\n")
            else:
                print("👤 No name set yet. You can set your name by saying something like 'My name is John' or 'Call me Sarah'.")
                print(f"📁 Will be stored in: {self.user_data_file}\n")
        elif cmd in ['/clear-name', '/cn']:
            old_name = self.user_name
            self.user_name = None
            self._save_user_data()
            if old_name:
                print(f"✅ Cleared your name '{old_name}'. You can set a new name anytime.\n")
            else:
                print("✅ No name was set to clear.\n")
        elif cmd in ['/weather'] and len(parts) >= 3:
            try:
                lat, lon = float(parts[1]), float(parts[2])
                print(f"🌤  Fetching weather data for ({lat}, {lon})...")
                weather_data = self.llm_client.get_weather_data(lat, lon)
                if "error" in weather_data:
                    print(f"❌ {weather_data['error']}\n")
                else:
                    print(f"✅ Weather data retrieved!")
                    print(f"📊 Data: {json.dumps(weather_data, indent=2)}\n")
            except ValueError:
                print("❌ Invalid coordinates. Use: /weather <latitude> <longitude>\n")
        elif cmd in ['/tools', '/t']:
            print("\n🔧 Available MCP Tools:")
            if self.tools_loaded:
                for tool in self.all_tools:
                    print(f"  📦 {tool.name}: {tool.description}")
            else:
                print("  ⚠  Tools not yet loaded. They will be loaded when MCP servers initialize.")
            print()
        elif cmd in ['/status', '/s']:
            print("\n📊 System Status:")
            print(f"  🔌 MCP Servers: {len(self.servers)} configured")
            print(f"  🔧 Tools Loaded: {'Yes' if self.tools_loaded else 'No'} ({len(self.all_tools)} tools)")
            print(f"  🤖 Available Providers: {len(self.llm_client.available_providers)}")
            current = self.llm_client.current_provider
            print(f"  🎯 Current Provider: {current.value if current else 'None selected'}")
            print(f"  🔄 Fallback Mode: {'Enabled' if self.llm_client.fallback_mode else 'Disabled'}")
            print()
        elif cmd in ['/fallback', '/f']:
            self.llm_client.fallback_mode = not self.llm_client.fallback_mode
            status = "enabled" if self.llm_client.fallback_mode else "disabled"
            print(f"\n🔄 Fallback mode {status}\n")
        elif cmd in ['/quit', '/exit', '/q']:
            print("\n👋 Goodbye!\n")
            return True
        else:
            print(f"❌ Unknown command: {cmd}. Type /help for available commands.\n")
        return False

    def show_help(self) -> None:
        print("\n" + "="*70)
        print("🤖 Multi-LLM Chat Assistant with MCP Tools")
        print("="*70)
        print("\n📋 Available Commands:")
        print("  /help, /h              - Show this help message")
        print("  /providers, /p         - List available LLM providers")
        print("  /switch <provider>     - Switch to different LLM provider")
        print("  /current, /c           - Show current LLM provider")
        print("  /name, /n              - Show your remembered name")
        print("  /clear-name, /cn       - Clear your remembered name")
        print("  /fallback, /f          - Toggle automatic fallback mode")
        print("  /weather <lat> <lon>   - Get weather data for coordinates")
        print("  /tools, /t             - List available MCP tools")
        print("  /status, /s            - Show system status")
        print("  /quit, /exit, /q       - Exit the application")
        print("\n🔧 Available LLM Providers:")
        for idx, provider in enumerate(self.llm_client.list_available_providers(), 1):
            current = " ← ACTIVE" if self.llm_client.current_provider and provider == self.llm_client.current_provider.value else ""
            print(f"  [{idx}] {provider}{current}")
        print("\n💡 Example Usage:")
        print("  /switch gemini         - Switch to Gemini API")
        print("  /switch groq           - Switch to Groq API")
        print("  /switch ollama_mistral - Switch to local Mistral")
        print("  /weather 37.7749 -122.4194 - Weather for San Francisco")
        print("  My name is John        - Set your name to John")
        print("  Call me Sarah          - Set your name to Sarah")
        fallback_status = "enabled" if self.llm_client.fallback_mode else "disabled"
        print(f"\n🔄 Fallback Mode: {fallback_status}")
        print("   (When enabled, tries other providers if selected one fails)")
        print("="*70 + "\n")


async def main() -> None:
    os.makedirs('images', exist_ok=True)
    logging.info("Created 'images' folder for snapshot storage.")
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = MultiLLMClient(config)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())