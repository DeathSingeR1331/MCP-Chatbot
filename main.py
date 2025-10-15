import asyncio
import json
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, Any
from enum import Enum

import requests
import urllib.parse
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import base64  # For decoding base64 image data
import subprocess

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

from test_notion import (
    test_comment_on_page,
    test_search_pages,
    test_list_databases,
    test_retrieve_comments,
    test_query_database,
    test_retrieve_page,
    test_get_page_blocks,
)  # reuse working functions

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
                        await asyncio.wait_for(self.session.__aexit__(None, None, None), timeout=3.0)
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
                        await asyncio.wait_for(self.stdio_context.__aexit__(None, None, None), timeout=2.0)
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
                logging.info(f"🎯 Auto-selected LLM provider: {target_provider.value}")
            else:
                return "❌ No LLM providers available. Set GROQ_API_KEY or GEMINI_API_KEY in your environment/.env."
        try:
            response = self._get_response_from_provider(messages, target_provider)
            if response:
                logging.info(f"✓ Response from {target_provider.value}")
                return response
        except Exception as e:
            error_msg = f"Provider {target_provider.value} error: {str(e)}"
            logging.error(error_msg)
            if self.fallback_mode:
                logging.info("🔄 Fallback mode enabled, trying other providers...")
                for fallback_provider in self.available_providers:
                    if fallback_provider == target_provider:
                        continue
                    try:
                        response = self._get_response_from_provider(messages, fallback_provider)
                        if response:
                            logging.info(f"✓ Fallback successful with {fallback_provider.value}")
                            return response
                    except Exception as fallback_error:
                        logging.warning(f"✗ Fallback provider {fallback_provider.value} also failed: {fallback_error}")
                        continue
            return f"❌ {error_msg}"
        return f"⚠️ Failed to get response from {target_provider.value}."

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

    async def cleanup_servers(self) -> None:
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        if cleanup_tasks:
            try:
                results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        server_name = self.servers[i].name if i < len(self.servers) else f"server_{i}"
                        if isinstance(result, asyncio.CancelledError):
                            logging.info(f"Note: Normal shutdown message for {server_name}: {result}")
                        else:
                            logging.warning(f"Warning during session cleanup for {server_name}: {result}")
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def chat(self, user_message: str) -> str:
        try:
            lowered = user_message.lower()
            if ("youtube" in lowered and "play" in lowered) or lowered.startswith("play "):
                song_query = self._extract_youtube_song_query(lowered)
                if song_query:
                    played = await self._play_youtube_song_with_retry(song_query)
                    return "✅ Playing on YouTube." if played else "❌ Failed to start playback on YouTube after all attempts."
            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools]) if self.all_tools else "No tools available."
            #
            # Build the system prompt that instructs the language model on how to choose and call tools.
            #
            # In earlier versions, the model would sometimes use general-purpose browser tools (like
            # ``browser_evaluate``) to try and fetch information from web pages when asked for news
            # headlines.  This led to confusing or incorrect outputs (e.g. returning a page title
            # instead of actual news articles).  To remedy this, the system message explicitly
            # instructs the model to ALWAYS use the dedicated ``get_news`` tool for any queries
            # requesting news, headlines, or the latest updates.  It also discourages the use of
            # browser tools for news-related queries.  These additional guidelines reduce the
            # likelihood that the model will misuse browser tools for tasks better handled by the
            # news API.
            system_message = f"""
You are a helpful assistant with access to these tools:

{tools_description}

CRITICAL: You MUST use the EXACT tool names listed above.  Common browser tools include:
- browser_navigate (to navigate to a URL)
- browser_click (to click on elements)
- browser_take_screenshot (to take screenshots)
- browser_close (to close the page)
- browser_resize (to resize browser window)
- browser_get_current_url (to get current URL)
- browser_get_page_title (to get page title)

WEATHER TOOL:
- get_weather (to get current weather for any city)
  Arguments: {{"city": "city_name"}} (e.g., "New York", "London", "Tokyo")

NEWS TOOL:
- get_news (to get latest news articles on any topic)
  Arguments: {{"query": "search_query", "max_results": 10, "language": "en"}} (e.g., "technology", "politics", "sports")

NEWS QUERIES:
- If the user asks for news, headlines, latest updates, or anything similar, you MUST use
  ``get_news``.  DO NOT use browser tools (like ``browser_evaluate`` or ``browser_navigate``) to
  scrape news pages.  Always call ``get_news`` with an appropriate ``query`` argument to fetch
  the latest articles.

Choose the appropriate tool based on the user's question.  If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, respond with the exact JSON object format below.  For
multiple tools or multi-step tasks, provide multiple JSON objects, each on a separate line.
Nothing else in the response:

{{
    "tool": "exact-tool-name-from-list-above",
    "arguments": {{
        "argument-name": "value"
    }}
}}

SPECIAL INSTRUCTIONS FOR MUSIC/VIDEO PLAYBACK:
- When a user asks to "play" a song/video, you MUST do TWO steps:
  1. First: Use ``browser_navigate`` to go to YouTube search results.
  2. Second: Use ``browser_click`` or evaluate JavaScript to click on the first video result.
- For YouTube searches, use: https://www.youtube.com/results?search_query=SONG_NAME
- After navigating, click on the first video link to actually play it.

For screenshot tools (``browser_take_screenshot``, ``puppeteer_screenshot``, etc.), always use
``images/`` as the prefix for the path (e.g. ``images/screenshot.png``) so the file is saved
into the images folder.  Use boolean values for the ``fullPage`` parameter (true/false, not
strings).

For navigation, use ``browser_navigate`` with the "url" argument.

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response.
2. Keep responses concise but informative.
3. Focus on the most relevant information.
4. Use appropriate context from the user's question.
5. Avoid simply repeating the raw data.

Please use only the tools that are explicitly defined above with their EXACT names.
"""
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
                        ok = await self._play_youtube_song_with_retry(query)
                        results.append("YouTube playback started" if ok else "Failed to start YouTube playback after all attempts")
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
                        result = f"Weather in {city}: {description}, Temperature: {temp}°C, Humidity: {humidity}%"
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
            is_success = any("executed successfully" in r for r in results) and not any("Error" in r for r in results)
            if is_success:
                return "✅ Task completed successfully."
            if any("Error" in r for r in results):
                return "❌ Task failed during tool execution."
            return "ℹ️ Tools executed."
        return llm_response

    async def _play_via_local_playwright(self, query: str) -> bool:
        """Enhanced local Playwright automation with better browser configuration."""
        try:
            # Escape the query for safe use in the code string
            escaped_query = query.replace('"', '\\"').replace("'", "\\'")
            
            code = f'''import asyncio
from playwright.async_api import async_playwright

VIDEO_QUERY = "{escaped_query}"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False, 
            args=[
                "--autoplay-policy=no-user-gesture-required",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--enable-automation",
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        ctx = await browser.new_context(
            permissions=["microphone","camera"],
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await ctx.new_page()
        
        # Navigate to search results
        search_url = "https://www.youtube.com/results?search_query=" + VIDEO_QUERY.replace(" ", "+")
        await page.goto(search_url)
        await page.wait_for_timeout(3000)
        
        # Try multiple selectors for video selection
        selectors = [
            'ytd-video-renderer a#thumbnail',
            'ytd-video-renderer a#video-title',
            'ytd-video-renderer h3 a',
            'a#video-title',
            'a#thumbnail'
        ]
        
        clicked = False
        for selector in selectors:
            try:
                element = page.locator(selector).first
                if element.count() > 0:
                    await element.scroll_into_view_if_needed()
                    await element.click()
                    clicked = True
                    break
            except Exception as e:
                print(f"Selector {{selector}} failed: {{e}}")
                continue
        
        if not clicked:
            print("No video found with any selector")
            return
        
        # Wait for video page to load
        await page.wait_for_selector("video", timeout=10000)
        await page.wait_for_timeout(2000)
        
        # Try to start playback
        try:
            # Click play button if visible
            play_button = page.locator('.ytp-play-button, .ytp-large-play-button')
            if play_button.count() > 0:
                await play_button.click()
            
            # Also try direct video play
            await page.evaluate("""
                () => {{
                    const video = document.querySelector('video');
                    if (video) {{
                        video.muted = false;
                        video.volume = 0.7;
                        video.play().catch(e => {{
                            console.log('Play failed, trying muted:', e);
                            video.muted = true;
                            video.play();
                        }});
                    }}
                }}
            """)
            
        except Exception as e:
            print(f"Playback attempt failed: {{e}}")
        
        await page.wait_for_timeout(5000)
        await browser.close()

asyncio.run(main())
'''
            proc = subprocess.Popen([sys.executable, "-c", code])
            return proc is not None
        except Exception as e:
            logging.error(f"Local Playwright spawn failed: {e}")
            return False

    async def _play_youtube_song_with_retry(self, query: str, max_retries: int = 3) -> bool:
        """YouTube automation with retry mechanism and comprehensive logging."""
        logging.info(f"🎵 Starting YouTube automation with retry for: {query}")
        
        for attempt in range(max_retries):
            logging.info(f"🔄 Attempt {attempt + 1}/{max_retries}")
            
            try:
                # Try the enhanced automation first
                if await self._play_youtube_song(query):
                    logging.info(f"✅ YouTube automation successful on attempt {attempt + 1}")
                    return True
                    
                # If enhanced automation fails, try local Playwright as fallback
                logging.info("🔄 Enhanced automation failed, trying local Playwright fallback...")
                if await self._play_via_local_playwright(query):
                    logging.info(f"✅ Local Playwright fallback successful on attempt {attempt + 1}")
                    return True
                    
            except Exception as e:
                logging.error(f"❌ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                logging.info(f"⏳ Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        logging.error(f"❌ All {max_retries} attempts failed for YouTube automation")
        return False

    async def _play_youtube_song(self, query: str) -> bool:
        """Enhanced YouTube playback automation with autoplay policy handling and comprehensive error handling."""
        try:
            logging.info(f"🎵 Starting enhanced YouTube automation for: {query}")
            search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
            
            for server in self.servers:
                if not server.session:
                    continue
                try:
                    tools = await server.list_tools()
                    tool_names = {t.name for t in tools}
                except Exception:
                    continue
                    
                # Enhanced Playwright automation
                if {"browser_navigate", "browser_evaluate"}.issubset(tool_names):
                    logging.info("🔍 Using Playwright for YouTube automation")
                    
                    # Step 1: Navigate with enhanced settings
                    logging.info("🌐 Navigating to YouTube search results...")
                    await server.execute_tool("browser_navigate", {"url": search_url})
                    
                    # Step 2: Wait for search results to load with dynamic waiting
                    logging.info("⏳ Waiting for search results to load...")
                    await asyncio.sleep(3)
                    
                    # Step 3: Enhanced video selection with better selectors and user interaction simulation
                    logging.info("🎯 Selecting first video with enhanced selectors...")
                    click_js = """
                    () => {
                      // Wait for results to be visible
                      const resultsContainer = document.querySelector('#contents, #primary, ytd-search');
                      if (!resultsContainer) {
                        console.log('Results container not found, waiting...');
                        return false;
                      }
                      
                      // Enhanced selectors for better compatibility
                      const selectors = [
                        'ytd-video-renderer a#video-title',
                        'ytd-video-renderer a#thumbnail',
                        'ytd-video-renderer h3 a',
                        'ytd-video-renderer .ytd-thumbnail a',
                        'ytd-video-renderer ytd-thumbnail a',
                        'a#video-title',
                        'a#thumbnail',
                        'ytd-video-renderer #video-title-link',
                        'ytd-video-renderer a[href*="/watch?v="]'
                      ];
                      
                      for (const selector of selectors) {
                        const link = document.querySelector(selector);
                        if (link && link.href && link.href.includes('/watch?v=')) {
                          console.log('Found video link with selector:', selector);
                          // Simulate user interaction with scroll and click
                          link.scrollIntoView({ behavior: 'smooth', block: 'center' });
                          
                          // Create a proper click event
                          const clickEvent = new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            button: 0
                          });
                          
                          // Dispatch the event
                          link.dispatchEvent(clickEvent);
                          
                          // Also try direct click as fallback
                          setTimeout(() => {
                            if (link.click) link.click();
                          }, 100);
                          
                          return true;
                        }
                      }
                      
                      // Fallback: find any video link and simulate user interaction
                      const allLinks = document.querySelectorAll('a[href*="/watch?v="]');
                      if (allLinks.length > 0) {
                        console.log('Using fallback link selection');
                        const firstLink = allLinks[0];
                        firstLink.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        
                        const clickEvent = new MouseEvent('click', {
                          bubbles: true,
                          cancelable: true,
                          view: window,
                          button: 0
                        });
                        
                        firstLink.dispatchEvent(clickEvent);
                        setTimeout(() => {
                          if (firstLink.click) firstLink.click();
                        }, 100);
                        
                        return true;
                      }
                      
                      console.log('No video links found');
                      return false;
                    }
                    """
                    
                    result = await server.execute_tool("browser_evaluate", {"function": click_js})
                    logging.info(f"🎯 Video selection result: {result}")
                    
                    if not result:
                        logging.warning("❌ Failed to select video, trying next server...")
                        continue
                        
                    # Step 4: Wait for video page to load
                    logging.info("⏳ Waiting for video page to load...")
                    await asyncio.sleep(4)
                    
                    # Step 5: Enhanced video player interaction with autoplay policy handling
                    logging.info("▶️ Attempting to play video with enhanced interaction...")
                    for attempt in range(15):  # Increased attempts
                        play_js = """
                        () => {
                          // Check if we're on a video page
                          const video = document.querySelector('video');
                          if (!video) {
                            console.log('Video element not found, attempt:', arguments[0] || 0);
                            return false;
                          }
                          
                          console.log('Video element found, attempting to play...');
                          
                          // Handle autoplay policy and user interaction
                          const playButton = document.querySelector('.ytp-play-button, .ytp-large-play-button, .ytp-big-mode .ytp-play-button');
                          if (playButton) {
                            const ariaLabel = playButton.getAttribute('aria-label') || '';
                            if (ariaLabel.includes('Play') || ariaLabel.includes('play')) {
                              console.log('Clicking play button');
                              
                              // Simulate user interaction on play button
                              const clickEvent = new MouseEvent('click', {
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                button: 0
                              });
                              
                              playButton.dispatchEvent(clickEvent);
                              playButton.click();
                              
                              return true;
                            }
                          }
                          
                          // Try to play video directly with comprehensive user gesture simulation
                          try {
                            console.log('Attempting direct video play...');
                            
                            // Simulate user interaction on video element
                            const videoClickEvent = new MouseEvent('click', {
                              bubbles: true,
                              cancelable: true,
                              view: window,
                              button: 0
                            });
                            
                            video.dispatchEvent(videoClickEvent);
                            
                            // Set video properties for better autoplay
                            video.muted = false;
                            video.volume = 0.7;
                            video.autoplay = true;
                            
                            // Create additional user gesture events
                            const touchEvent = new TouchEvent('touchstart', {
                              bubbles: true,
                              cancelable: true,
                              touches: [new Touch({
                                identifier: 1,
                                target: video,
                                clientX: 100,
                                clientY: 100
                              })]
                            });
                            
                            video.dispatchEvent(touchEvent);
                            
                            // Attempt to play with promise handling
                            const playPromise = video.play();
                            if (playPromise !== undefined) {
                              playPromise.then(() => {
                                console.log('✅ Video started playing successfully');
                                return true;
                              }).catch(error => {
                                console.log('❌ Play failed:', error.message);
                                
                                // Try muted play as fallback
                                video.muted = true;
                                video.play().then(() => {
                                  console.log('✅ Video started playing (muted)');
                                  // Unmute after a short delay
                                  setTimeout(() => {
                                    video.muted = false;
                                  }, 1000);
                                }).catch(mutedError => {
                                  console.log('❌ Muted play also failed:', mutedError.message);
                                });
                              });
                            }
                            
                            // Check if video is playing
                            const isPlaying = !video.paused && !video.ended && video.readyState > 2;
                            console.log('Video state check:', {
                              paused: video.paused,
                              ended: video.ended,
                              readyState: video.readyState,
                              isPlaying: isPlaying,
                              currentTime: video.currentTime,
                              duration: video.duration
                            });
                            
                            return isPlaying;
                            
                          } catch (error) {
                            console.log('❌ Video play error:', error.message);
                            return false;
                          }
                        }
                        """
                        
                        result = await server.execute_tool("browser_evaluate", {"function": play_js})
                        logging.info(f"▶️ Play attempt {attempt + 1}/15: {result}")
                        
                        if result:
                            # Verify video is actually playing with comprehensive check
                            logging.info("🔍 Verifying video playback...")
                            verify_js = """
                            () => {
                              const video = document.querySelector('video');
                              if (!video) {
                                console.log('❌ No video element for verification');
                                return false;
                              }
                              
                              // Comprehensive video state check
                              const isPlaying = !video.paused && !video.ended && video.readyState > 2;
                              const hasDuration = video.duration > 0;
                              const hasCurrentTime = video.currentTime > 0;
                              const isLoaded = video.readyState >= 3; // HAVE_FUTURE_DATA
                              
                              const videoState = {
                                isPlaying,
                                hasDuration,
                                hasCurrentTime,
                                isLoaded,
                                currentTime: video.currentTime,
                                duration: video.duration,
                                readyState: video.readyState,
                                paused: video.paused,
                                ended: video.ended,
                                muted: video.muted,
                                volume: video.volume
                              };
                              
                              console.log('📊 Video verification state:', videoState);
                              
                              // Video is considered successfully playing if:
                              // 1. Not paused and not ended
                              // 2. Has duration (loaded metadata)
                              // 3. Has current time (actually playing)
                              // 4. Ready state indicates data is available
                              const success = isPlaying && hasDuration && hasCurrentTime && isLoaded;
                              
                              if (success) {
                                console.log('✅ Video verification successful - video is playing!');
                              } else {
                                console.log('❌ Video verification failed - video not playing properly');
                              }
                              
                              return success;
                            }
                            """
                            
                            await asyncio.sleep(2)  # Wait for video to start
                            verification = await server.execute_tool("browser_evaluate", {"function": verify_js})
                            
                            if verification:
                                logging.info("✅ Video successfully started playing and verified!")
                                return True
                            else:
                                logging.warning("⚠️ Video started but verification failed, continuing attempts...")
                        
                        await asyncio.sleep(1)  # Wait between attempts
                    
                    logging.warning("❌ Failed to start video playback after all attempts")
                    return False
                    
                # Enhanced Puppeteer automation
                elif {"puppeteer_navigate", "puppeteer_evaluate"}.issubset(tool_names):
                    logging.info("🔍 Using Puppeteer for YouTube automation")
                    
                    # Similar enhanced logic for Puppeteer
                    await server.execute_tool("puppeteer_navigate", {"url": search_url})
                    await asyncio.sleep(3)
                    
                    # Enhanced Puppeteer click logic
                    click_js = """
                    () => {
                      const selectors = [
                        'ytd-video-renderer a#video-title',
                        'ytd-video-renderer a#thumbnail',
                        'ytd-video-renderer h3 a',
                        'a#video-title',
                        'a#thumbnail'
                      ];
                      
                      for (const selector of selectors) {
                        const link = document.querySelector(selector);
                        if (link && link.href && link.href.includes('/watch?v=')) {
                          console.log('Puppeteer found link with selector:', selector);
                          link.scrollIntoView({ behavior: 'smooth', block: 'center' });
                          link.click();
                          return true;
                        }
                      }
                      
                      const allLinks = document.querySelectorAll('a[href*="/watch?v="]');
                      if (allLinks.length > 0) {
                        allLinks[0].click();
                        return true;
                      }
                      
                      return false;
                    }
                    """
                    
                    result = await server.execute_tool("puppeteer_evaluate", {"script": click_js})
                    logging.info(f"🎯 Puppeteer video selection result: {result}")
                    
                    if result:
                        await asyncio.sleep(4)
                        
                        # Enhanced Puppeteer play logic
                        for attempt in range(15):
                            play_js = """
                            () => {
                              const video = document.querySelector('video');
                              if (!video) return false;
                              
                              const playButton = document.querySelector('.ytp-play-button');
                              if (playButton) {
                                playButton.click();
                                return true;
                              }
                              
                              try {
                                video.muted = false;
                                video.volume = 0.7;
                                const playPromise = video.play();
                                if (playPromise !== undefined) {
                                  playPromise.catch(() => {
                                    video.muted = true;
                                    video.play();
                                  });
                                }
                                return !video.paused;
                              } catch (e) {
                                return false;
                              }
                            }
                            """
                            
                            result = await server.execute_tool("puppeteer_evaluate", {"script": play_js})
                            if result:
                                await asyncio.sleep(2)
                                verify_js = """
                                () => {
                                  const video = document.querySelector('video');
                                  if (!video) return false;
                                  return !video.paused && !video.ended && video.readyState > 2 && video.currentTime > 0;
                                }
                                """
                                verification = await server.execute_tool("puppeteer_evaluate", {"script": verify_js})
                                if verification:
                                    logging.info("✅ Puppeteer video successfully started playing!")
                                    return True
                            
                            await asyncio.sleep(1)
                    
                    logging.warning("❌ Puppeteer failed to start video playback")
                    return False
                    
            logging.error("❌ No suitable browser automation tools found")
            return False
            
        except Exception as e:
            logging.error(f"❌ YouTube playback automation failed: {e}")
            import traceback
            logging.error(f"Full error traceback: {traceback.format_exc()}")
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

    # Removed duplicate _play_youtube_song definition to avoid infinite recursion

    async def start(self) -> None:
        try:
            print("\n" + "="*70)
            print("🚀 Starting Multi-LLM Chat Assistant...")
            print("="*70)
            print("\n🔌 Initializing MCP servers...")
            initialized_servers = 0
            for server in self.servers:
                try:
                    await server.initialize()
                    initialized_servers += 1
                    print(f"  ✓ Server '{server.name}' initialized")
                except Exception as e:
                    logging.error(f"  ✗ Failed to initialize server '{server.name}': {e}")
            if initialized_servers == 0:
                print("\n⚠  No MCP servers were initialized. Continuing without tools...")
            print("\n🔧 Loading tools...")
            self.all_tools = []
            for server in self.servers:
                if server.session:
                    try:
                        tools = await server.list_tools()
                        self.all_tools.extend(tools)
                        print(f"  ✓ Loaded {len(tools)} tools from '{server.name}'")
                    except Exception as e:
                        logging.warning(f"  ✗ Could not load tools from '{server.name}': {e}")
            self.tools_loaded = len(self.all_tools) > 0
            print(f"\n📦 Total tools available: {len(self.all_tools)}")
            print(f"\n🔍 Available tools from each server:")
            for server in self.servers:
                if server.session:
                    try:
                        tools = await server.list_tools()
                        tool_names = [tool.name for tool in tools]
                        print(f"  📦 {server.name}: {tool_names}")
                    except Exception as e:
                        print(f"  ❌ {server.name}: Error listing tools - {e}")
            print("\n🤖 SELECT YOUR LLM PROVIDER:")
            print("="*70)
            available_providers = self.llm_client.list_available_providers()
            if not available_providers:
                print("❌ No LLM providers available! Please check your API keys and installations.")
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
                    choice = input(f"\n👉 Enter your choice (1-{len(available_providers)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_providers):
                        selected_provider = available_providers[choice_idx]
                        if self.llm_client.switch_provider(selected_provider):
                            print(f"\n✅ Selected: {selected_provider}")
                            current_provider = self.llm_client.current_provider
                            print(f"🎯 Current provider: {current_provider.value if current_provider else 'None'}")
                            break
                        else:
                            print("❌ Failed to switch provider. Please try again.")
                    else:
                        print(f"❌ Invalid choice. Please enter a number between 1 and {len(available_providers)}.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\n\n👋 Exiting...")
                    return
            print("\n💡 Type /help for commands or start chatting!")
            print("="*70)
            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools]) if self.all_tools else "No tools available."
            #
            # Construct the system message for the interactive chat loop.  In addition to the core
            # instructions, we explicitly document the ``get_news`` tool and clarify that it must be
            # used for all news-related questions.  This prevents the LLM from misusing browser
            # tools when the user is asking for headlines or latest updates.
            system_message = (
                "You are a helpful assistant with access to these tools: \n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.\n\n"
                "NEWS QUERIES:\n"
                "- If the user asks for news, headlines, latest updates, sports news, or anything similar, you MUST use\n"
                "  the `get_news` tool described below. DO NOT use browser tools (like `browser_evaluate` or\n"
                "  `browser_navigate`) to scrape web pages for news. Always call `get_news` with an appropriate\n"
                "  `query` argument to fetch the latest articles.\n\n"
                "IMPORTANT: When you need to use a tool, respond with the exact JSON object format below. For multiple\n"
                "tools or multi-step tasks, provide multiple JSON objects, each on a separate line. Nothing else in the response:\n\n"
                "{\n"
                "    \"tool\": \"tool-name\",\n"
                "    \"arguments\": {\n"
                "        \"argument-name\": \"value\"\n"
                "    }\n"
                "}\n\n"
                "For screenshot tools (browser_take_screenshot, puppeteer_screenshot, etc.), always use a path\n"
                "starting with 'images/' (e.g., 'images/screenshot.png') to save in the images folder. Use boolean\n"
                "values for fullPage parameter (true/false, not \"true\"/\"false\").\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
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