"""
Test script to verify LLM API connections
Run this to diagnose which APIs are working
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("="*70)
print("🧪 Testing LLM API Connections")
print("="*70)

# Test Groq
print("\n[1] Testing Groq API...")
try:
    from groq import Groq
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("  ❌ Groq API key not set in .env file")
    else:
        print(f"  ✓ Groq library installed")
        print(f"  ✓ API key found: {groq_key[:20]}...")
        
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'Groq works!'"}],
            model="llama-3.1-8b-instant",
            max_tokens=50
        )
        print(f"  ✅ Groq API working! Response: {response.choices[0].message.content}")
except ImportError:
    print("  ❌ Groq library not installed. Run: pip install groq")
except Exception as e:
    print(f"  ❌ Groq API Error: {e}")

# Test Gemini
print("\n[2] Testing Gemini API...")
try:
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("  ❌ Gemini API key not set in .env file")
    else:
        print(f"  ✓ Gemini library installed")
        print(f"  ✓ API key found: {gemini_key[:20]}...")
        
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Say 'Gemini works!'")
        print(f"  ✅ Gemini API working! Response: {response.text}")
except ImportError:
    print("  ❌ Gemini library not installed. Run: pip install google-generativeai")
except Exception as e:
    print(f"  ❌ Gemini API Error: {e}")

# Test Ollama
print("\n[3] Testing Ollama (Mistral)...")
try:
    import ollama
    print(f"  ✓ Ollama library installed")
    
    response = ollama.chat(
        model='mistral:latest',
        messages=[{'role': 'user', 'content': "Say 'Ollama Mistral works!'"}]
    )
    print(f"  ✅ Ollama Mistral working! Response: {response['message']['content']}")
except ImportError:
    print("  ❌ Ollama library not installed. Run: pip install ollama")
except Exception as e:
    print(f"  ❌ Ollama Error: {e}")
    print("  💡 Make sure Ollama is running and mistral:latest is pulled")

# Test Ollama Qwen
print("\n[4] Testing Ollama (Qwen 2.5)...")
try:
    import ollama
    response = ollama.chat(
        model='qwen2.5:latest',
        messages=[{'role': 'user', 'content': "Say 'Ollama Qwen works!'"}]
    )
    print(f"  ✅ Ollama Qwen working! Response: {response['message']['content']}")
except ImportError:
    print("  ❌ Ollama library not installed. Run: pip install ollama")
except Exception as e:
    print(f"  ❌ Ollama Qwen Error: {e}")
    print("  💡 Make sure qwen2.5:latest is pulled: ollama pull qwen2.5:latest")

# Test Weather API (Open-Meteo - Free!)
print("\n[5] Testing Weather API (Open-Meteo)...")
try:
    import requests
    print(f"  ✓ Requests library available")
    print(f"  ✓ Using Open-Meteo (no API key needed!)")
    
    # Test with San Francisco coordinates
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': 37.7749,
        'longitude': -122.4194,
        'current': 'temperature_2m,apparent_temperature,is_day,wind_speed_10m,relative_humidity_2m',
        'hourly': 'temperature_2m,relative_humidity_2m',
        'models': 'best_match'
    }
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        temp = data.get('current', {}).get('temperature_2m', 'N/A')
        humidity = data.get('current', {}).get('relative_humidity_2m', 'N/A')
        print(f"  ✅ Weather API working! Temp: {temp}°C, Humidity: {humidity}%")
    else:
        print(f"  ❌ Weather API Error: Status {response.status_code}")
        print(f"     Response: {response.text[:200]}")
except Exception as e:
    print(f"  ❌ Weather API Error: {e}")

print("\n" + "="*70)
print("📊 Test Summary")
print("="*70)
print("\n✅ = Working  |  ❌ = Not working  |  💡 = Action needed")
print("\nIf any tests failed, check:")
print("  1. API keys are correctly set in .env file")
print("  2. Required packages are installed (see requirements.txt)")
print("  3. Ollama is running (for local models)")
print("  4. Internet connection is active (for API calls)")
print("="*70 + "\n")

