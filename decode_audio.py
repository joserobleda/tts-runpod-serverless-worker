import base64
import os
import json

def decode_and_save_audio(base64_string, output_filename="output_audio.wav"):
    """Decode base64 audio and save as WAV file"""
    try:
        # Clean the base64 string (remove whitespace and newlines)
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Decode the base64 string
        audio_bytes = base64.b64decode(base64_string)
        
        # Write to file
        with open(output_filename, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"✅ Audio saved as {output_filename}")
        print(f"📁 File size: {len(audio_bytes)} bytes")
        print(f"🎵 You can now play it with any audio player!")
        
        # Try to open it automatically (works on most systems)
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_filename)
            elif os.name == 'posix':  # macOS/Linux
                os.system(f'open "{output_filename}"')  # macOS
                # os.system(f'xdg-open "{output_filename}"')  # Linux alternative
        except:
            print("💡 Open the file manually with your audio player")
            
    except Exception as e:
        print(f"❌ Error decoding audio: {e}")

def read_base64_from_file(filename):
    """Read base64 string from various file formats"""
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
        
        # Try to parse as JSON first (in case it's the full API response)
        try:
            data = json.loads(content)
            # Look for audio field in the JSON
            if isinstance(data, dict):
                if 'output' in data and 'audio' in data['output']:
                    return data['output']['audio']
                elif 'audio' in data:
                    return data['audio']
            print("❌ Could not find 'audio' field in JSON")
            return None
        except json.JSONDecodeError:
            # Not JSON, treat as plain text base64
            return content
            
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

if __name__ == "__main__":
    # List of possible input files to look for
    possible_files = [
        "audio.txt",           # Plain base64 text file
        "audio_base64.txt",    # Plain base64 text file  
        "response.json",       # Full API response JSON
        "audio_response.json", # Full API response JSON
        "output.json"          # Full API response JSON
    ]
    
    print("🔍 Looking for audio files...")
    
    # Check which files exist
    existing_files = [f for f in possible_files if os.path.exists(f)]
    
    if not existing_files:
        print("📝 No audio files found. Please create one of these files:")
        print("   • audio.txt - containing just the base64 string")
        print("   • response.json - containing the full API response")
        print("")
        print("💡 Example audio.txt content:")
        print("   UklGRiSCBQBXQVZFZm10IBAAAAABAAEAwF0...")
        print("")
        print("💡 Example response.json content:")
        print('   {"output": {"audio": "UklGRiSCBQBXQVZF..."}}')
        exit(1)
    
    # Use the first existing file
    input_file = existing_files[0]
    print(f"📂 Reading from: {input_file}")
    
    # Read the base64 string
    base64_string = read_base64_from_file(input_file)
    
    if base64_string:
        print(f"✅ Found base64 string ({len(base64_string)} characters)")
        decode_and_save_audio(base64_string)
    else:
        print("❌ Could not extract base64 string from file") 