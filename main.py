import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Any
from enum import Enum

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import base64  # New: For decoding base64 image data

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
    test_get_page_blocks
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
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "08507b919c3b0d939054afb577c46082")

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
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
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
        
        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )
        
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
        delay: float = 1.0
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

        # Enhanced: For screenshot tools, ensure proper path handling and parameter types
        screenshot_tools = ['browser_screenshot', 'browser_take_screenshot', 'puppeteer_screenshot', 'puppeteer_take_screenshot']
        if tool_name in screenshot_tools:
            # Fix parameter types for screenshot tools
            if 'fullPage' in arguments and isinstance(arguments['fullPage'], str):
                arguments['fullPage'] = arguments['fullPage'].lower() == 'true'
                logging.info(f"[{self.name}] Converted fullPage string to boolean: {arguments['fullPage']}")
            
            if 'path' not in arguments:
                # Ensure images directory exists
                os.makedirs('images', exist_ok=True)
                timestamp = int(asyncio.get_event_loop().time())
                arguments['path'] = f"images/screenshot_{tool_name}_{timestamp}.png"
                logging.info(f"[{self.name}] Added default image path for {tool_name}: {arguments['path']}")
            elif not arguments['path'].startswith('images/'):
                # If path doesn't start with images/, prepend it
                arguments['path'] = f"images/{arguments['path']}"
                logging.info(f"[{self.name}] Modified path to save in images folder: {arguments['path']}")

        logging.info(f"[{self.name}] Attempting to execute tool '{tool_name}' with arguments: {arguments}")
        
        attempt = 0
        while attempt < retries:
            try:
                supports_progress = (
                    self.capabilities 
                    and 'progress' in self.capabilities
                )

                logging.info(f"[{self.name}] Progress tracking supported: {supports_progress}")
                logging.info(f"[{self.name}] Server capabilities: {self.capabilities}")

                if supports_progress:
                    logging.info(f"[{self.name}] Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(
                        tool_name, 
                        arguments
                    )
                else:
                    logging.info(f"[{self.name}] Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)

                logging.info(f"[{self.name}] Tool '{tool_name}' executed successfully with result: {result}")

                # New: Save screenshot if it's a screenshot tool and result contains image data
                if tool_name in screenshot_tools and result and hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'mimeType') and content.mimeType == 'image/png' and hasattr(content, 'data'):
                            # Decode base64 image data and save to file
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
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
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
                # Enhanced: Note for path in screenshot tools
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
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE and self.config.gemini_api_key and genai:
            try:
                # Configure Gemini API
                genai.configure(api_key=self.config.gemini_api_key)  # type: ignore
            except Exception as e:
                logging.warning(f"Failed to configure Gemini: {e}")
        
        # Initialize Groq if available
        if GROQ_AVAILABLE and self.config.groq_api_key:
            self.groq_client = Groq(api_key=self.config.groq_api_key)
        
        # Ollama removed - only using Groq and Gemini

        # Auto-select a sensible default provider if available so users don't need /switch
        try:
            if LLMProvider.GROQ in self.available_providers:
                self.current_provider = LLMProvider.GROQ
            elif LLMProvider.GEMINI in self.available_providers:
                self.current_provider = LLMProvider.GEMINI
        except Exception:
            # Non-fatal: leave provider unset; get_response will handle gracefully
            pass

    def get_response(self, messages: List[Dict[str, str]], provider: Optional[LLMProvider] = None) -> str:
        """Get a response from the selected LLM provider.
        
        Args:
            messages: A list of message dictionaries.
            provider: Specific provider to use, or use current_provider if None.
            
        Returns:
            The LLM's response as a string.
        """
        # Use the specified provider or current provider; if none, auto-select first available
        target_provider = provider or self.current_provider
        if not target_provider:
            if self.available_providers:
                self.current_provider = self.available_providers[0]
                target_provider = self.current_provider
                logging.info(f"🎯 Auto-selected LLM provider: {target_provider.value}")
            else:
                return (
                    "❌ No LLM providers available. Set GROQ_API_KEY or GEMINI_API_KEY in your environment/.env."
                )
        
        # Try the selected provider
        try:
            response = self._get_response_from_provider(messages, target_provider)
            if response:
                logging.info(f"✓ Response from {target_provider.value}")
                return response
        except Exception as e:
            error_msg = f"Provider {target_provider.value} error: {str(e)}"
            logging.error(error_msg)
            
            # If fallback mode is enabled, try other providers
            if self.fallback_mode:
                logging.info("🔄 Fallback mode enabled, trying other providers...")
                for fallback_provider in self.available_providers:
                    if fallback_provider == target_provider:
                        continue  # Skip the failed provider
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
        """Get response from a specific provider.
        
        Args:
            messages: A list of message dictionaries.
            provider: The LLM provider to use.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            Exception: If the provider fails.
        """
        if provider == LLMProvider.GEMINI:
            return self._get_gemini_response(messages)
        elif provider == LLMProvider.GROQ:
            return self._get_groq_response(messages)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_gemini_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from Gemini API."""
        if not GEMINI_AVAILABLE or not self.config.gemini_api_key:
            raise Exception("Gemini API not available")
        
        # Convert messages to Gemini format
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
        """Get response from Groq API."""
        if not GROQ_AVAILABLE or not self.config.groq_api_key:
            raise Exception("Groq API not available")
        
        try:
            # Try current recommended models in order
            models = [
                "llama-3.1-8b-instant",      # Fast and efficient
                "llama-3.3-70b-versatile",   # Latest Llama 3.3
                "llama-3.1-70b-versatile"    # Alternative
            ]
            
            for model in models:
                try:
                    # Convert messages to proper format for Groq
                    groq_messages = []
                    for msg in messages:
                        groq_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    response = self.groq_client.chat.completions.create(
                        messages=groq_messages,
                        model=model,
                        temperature=0.7,
                        max_tokens=4096,
                        top_p=1,
                        stream=False
                    )
                    return response.choices[0].message.content or "No response generated"
                except Exception as model_error:
                    if model == models[-1]:  # Last model
                        raise model_error
                    logging.warning(f"Model {model} failed: {model_error}. Trying next...")
                    continue
            
            # This should never be reached, but just in case
            return "No response generated"
        except Exception as e:
            raise Exception(f"All Groq models failed: {e}")

    # Ollama methods removed - only using Groq and Gemini

    def get_weather_data(self, lat: float, lon: float, exclude: str = "minutely,hourly") -> Dict[str, Any]:
        """Get weather data from Open-Meteo API (Free).
        
        Args:
            lat: Latitude coordinate.
            lon: Longitude coordinate.
            exclude: Not used (kept for compatibility).
            
        Returns:
            Weather data dictionary.
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,apparent_temperature,is_day,wind_speed_10m,wind_direction_10m,wind_gusts_10m,rain,showers,precipitation,snowfall,surface_pressure,cloud_cover,weather_code,pressure_msl,relative_humidity_2m',
            'hourly': 'temperature_2m,relative_humidity_2m,rain,showers,precipitation,visibility,weather_code,is_day,sunshine_duration',
            'models': 'best_match'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Weather API error: {e}")
            return {"error": f"Failed to fetch weather data: {e}"}

    def list_available_providers(self) -> List[str]:
        """List available LLM providers."""
        return [provider.value for provider in self.available_providers]

    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a specific LLM provider.
        
        Args:
            provider_name: Name of the provider to switch to.
            
        Returns:
            True if switch was successful, False otherwise.
        """
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
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def chat(self, user_message: str) -> str:
        """Process a single user message and return the response.
        
        Args:
            user_message: The user's message to process.
            
        Returns:
            The assistant's response.
        """
        try:
            # Build system message
            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools]) if self.all_tools else "No tools available."
            
            system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}

CRITICAL: You MUST use the EXACT tool names listed above. Common browser tools include:
- browser_navigate (to navigate to a URL)
- browser_click_element (to click on elements)
- browser_take_screenshot (to take screenshots)
- browser_close (to close the page)
- browser_resize (to resize browser window)
- browser_get_current_url (to get current URL)
- browser_get_page_title (to get page title)

WEATHER TOOL:
- get_weather (to get current weather for any city)
  Arguments: {{"city": "city_name"}} (e.g., "New York", "London", "Tokyo")

Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, respond with the exact JSON object format below. For multiple tools or multi-step tasks, provide multiple JSON objects, each on a separate line. Nothing else in the response:

{{
    "tool": "exact-tool-name-from-list-above",
    "arguments": {{
        "argument-name": "value"
    }}
}}

SPECIAL INSTRUCTIONS FOR MUSIC/VIDEO PLAYBACK:
- When user asks to "play" a song/video, you MUST do TWO steps:
  1. First: Use browser_navigate to go to YouTube search results
  2. Second: Use browser_click_element to click on the first video result
- For YouTube searches, use: https://www.youtube.com/results?search_query=SONG_NAME
- After navigating, click on the first video link to actually play it

For screenshot tools (browser_take_screenshot, puppeteer_screenshot, etc.), always use a path starting with 'images/' (e.g., 'images/screenshot.png') to save in the images folder. Use boolean values for fullPage parameter (true/false, not "true"/"false").

For navigation, use browser_navigate with the "url" argument.

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above with their EXACT names."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            # Get LLM response
            llm_response = self.llm_client.get_response(messages)
            
            # Process the response and execute tools if needed
            result = await self.process_llm_response(llm_response)
            
            if result != llm_response:
                # Tool was executed, get final response
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "system", "content": result})
                
                final_response = self.llm_client.get_response(messages)
                return final_response
            else:
                # No tool execution, return original response
                return llm_response
                
        except Exception as e:
            logging.error(f"Error in chat method: {e}")
            return f"Error processing your request: {str(e)}"

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.
        
        Args:
            llm_response: The response from the LLM.
            
        Returns:
            The result of tool execution or the original response.
        """
        import json
        import re

        # Split response into individual JSON objects
        tool_calls = []
        try:
            # Extract JSON objects from the response (handles multiple JSON blocks)
            json_strings = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response.replace('\n', ''))
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

        # If no tool calls found, try single JSON parsing for backward compatibility
        if not tool_calls:
            try:
                tool_call = json.loads(llm_response)
                if "tool" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                return llm_response

        # Execute tool calls sequentially
        results = []
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call["tool"]
            arguments = tool_call["arguments"]
            logging.info(f"Executing tool {i+1}/{len(tool_calls)}: {tool_name}")
            logging.info(f"With arguments: {arguments}")
            
            tool_executed = False
            
            # Handle weather tool directly
            if tool_name == "get_weather":
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
            # Combine results and send back to LLM for final response
            combined_result = "\n".join(results)
            logging.info(f"All tool executions completed. Results: {combined_result}")
            
            # Create a summary message for the LLM
            summary_messages = [
                {
                    "role": "system", 
                    "content": f"Tool execution results: {combined_result}\n\nPlease provide a conversational summary of what was accomplished. If the tools executed successfully, confirm the actions were completed. If there were errors, explain what went wrong."
                },
                {
                    "role": "user", 
                    "content": "Please summarize the tool execution results in a conversational manner."
                }
            ]
            
            try:
                final_response = self.llm_client.get_response(summary_messages)
                return final_response
            except Exception as e:
                logging.error(f"Error getting final response from LLM: {e}")
                return f"Tools executed. Results: {combined_result}"
        
        return llm_response

    def show_help(self) -> None:
        """Show available commands and options."""
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

    async def _handle_command(self, command: str) -> bool:
        """Handle special commands.
        
        Args:
            command: The command string to handle.
            
        Returns:
            True if should exit, False otherwise.
        """
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

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            # Show welcome message
            print("\n" + "="*70)
            print("🚀 Starting Multi-LLM Chat Assistant...")
            print("="*70)
            
            # Initialize MCP servers
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
            
            # Load tools from initialized servers
            print("\n🔧 Loading tools...")
            self.all_tools = []
            for server in self.servers:
                if server.session:  # Only try if server initialized successfully
                    try:
                        tools = await server.list_tools()
                        self.all_tools.extend(tools)
                        print(f"  ✓ Loaded {len(tools)} tools from '{server.name}'")
                    except Exception as e:
                        logging.warning(f"  ✗ Could not load tools from '{server.name}': {e}")
            
            self.tools_loaded = len(self.all_tools) > 0
            print(f"\n📦 Total tools available: {len(self.all_tools)}")
            
            # Debug: Show available tools from each server
            print(f"\n🔍 Available tools from each server:")
            for server in self.servers:
                if server.session:
                    try:
                        tools = await server.list_tools()
                        tool_names = [tool.name for tool in tools]
                        print(f"  📦 {server.name}: {tool_names}")
                    except Exception as e:
                        print(f"  ❌ {server.name}: Error listing tools - {e}")
            
            # LLM Provider Selection
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
                    "ollama_qwen": "Ollama - Qwen 2.5 Latest (Local)"
                }
                print(f"  [{idx}] {provider_display.get(provider, provider)}")
            
            print("="*70)
            
            # Get user selection
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
            
            # Build system message (FIXED: Allow multiple tool calls)
            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools]) if self.all_tools else "No tools available."
            
            system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, respond with the exact JSON object format below. For multiple tools or multi-step tasks, provide multiple JSON objects, each on a separate line. Nothing else in the response:

{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

For screenshot tools (browser_take_screenshot, puppeteer_screenshot, etc.), always use a path starting with 'images/' (e.g., 'images/screenshot.png') to save in the images folder. Use boolean values for fullPage parameter (true/false, not "true"/"false").

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        should_exit = await self._handle_command(user_input)
                        if should_exit:
                            break
                        continue
                    
                    # Regular chat
                    messages.append({"role": "user", "content": user_input})
                    
                    llm_response = self.llm_client.get_response(messages)

                    assistant_output = llm_response

                    try:
                        data = json.loads(assistant_output)
                        if "tool" in data:
                            tool_name = data["tool"]
                            args = data.get("arguments", {})
                            
                            # Handle Notion API operations
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


async def main() -> None:
    """Initialize and run the chat session."""
    # New: Create images folder for snapshots
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