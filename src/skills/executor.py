# src/skills/executor.py
import subprocess
import webbrowser
import platform
import os
import urllib.parse
from datetime import datetime

# --- App Name Resolution ---
def get_official_app_name(app_name: str, os_name: str) -> str:
    """Resolves aliases to official application names."""
    if os_name == "Darwin":  # macOS
        app_name_map = {
            "chrome": "Google Chrome", "google chrome": "Google Chrome", "brave": "Brave Browser",
            "brave browser": "Brave Browser", "firefox": "Firefox", "safari": "Safari",
            "vscode": "Visual Studio Code", "vs code": "Visual Studio Code", "terminal": "Terminal",
            "iterm": "iTerm", "notes": "Notes", "calendar": "Calendar", "mail": "Mail",
            "settings": "System Settings", "system settings": "System Settings",
            "preferences": "System Settings", "system preferences": "System Settings",
            "spotify": "Spotify", "netflix": "Netflix", "apple tv": "TV", "tv": "TV", "music": "Music",
        }
        return app_name_map.get(app_name.lower(), app_name)
    # Add similar maps for Windows and Linux here
    return app_name

# --- Skill Functions ---

def open_app(app_name: str):
    """
    Tries to open an application locally. If it fails, it falls back
    to opening a known website or performing a web search.
    """
    os_name = platform.system()
    official_app_name = get_official_app_name(app_name, os_name)
    print(f"Attempting to open '{official_app_name}' on {os_name}...")
    try:
        if os_name == "Darwin":
            subprocess.run(["open", "-a", official_app_name], check=True)
            print(f"Successfully launched {official_app_name}.")
        elif os_name == "Windows":
            os.startfile(f"{official_app_name}.exe")
            print(f"Successfully launched {official_app_name}.")
        elif os_name == "Linux":
            subprocess.run([official_app_name.lower()], check=True)
            print(f"Successfully launched {official_app_name}.")
    except Exception as e:
        print(f"Error opening local app '{official_app_name}': {e}. Falling back to web.")
        website_map = {
            "amazon": "primevideo.com", "amazon prime": "primevideo.com", "prime video": "primevideo.com",
            "youtube": "youtube.com", "gmail": "gmail.com", "google drive": "drive.google.com",
        }
        url_guess = website_map.get(app_name.lower())
        if url_guess:
            open_url(url_guess)
        else:
            print(f"Don't have a specific website for '{app_name}'. Searching the web instead.")
            search_web(app_name)

def close_app(app_name: str):
    """Closes an application."""
    os_name = platform.system()
    official_app_name = get_official_app_name(app_name, os_name)
    print(f"Attempting to close '{official_app_name}' on {os_name}...")
    try:
        if os_name == "Darwin":
            command = f'osascript -e \'quit app "{official_app_name}"\''
            subprocess.run(command, shell=True, check=True)
        elif os_name == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", f"{official_app_name}.exe"], check=True)
        elif os_name == "Linux":
            subprocess.run(["pkill", official_app_name.lower()], check=True)
        print(f"Successfully closed {official_app_name}.")
    except Exception as e:
        print(f"Error closing app '{official_app_name}': {e}")

def open_url(url: str):
    """Opens a URL in the default browser."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        print(f"Opening URL: {url}")
        webbrowser.open(url)
    except Exception as e:
        print(f"Error opening URL '{url}': {e}")

def search_web(query: str):
    """Performs a web search."""
    try:
        print(f"Searching web for: '{query}'")
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        webbrowser.open(search_url)
    except Exception as e:
        print(f"Error performing web search for '{query}': {e}")

def create_note(content: str):
    """Creates a new text file with the given content inside project/notes/."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"note_{timestamp}.txt"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        notes_dir = os.path.join(base_dir, "../../notes") # Go up two levels from src/skills/
        if not os.path.exists(notes_dir):
            os.makedirs(notes_dir)
        filepath = os.path.join(notes_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Note saved at: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"Error creating note: {e}")

def play_media(media_title: str, platform: str = None):
    """Plays media, intelligently searching on YouTube or Spotify."""
    search_platform = "youtube"
    if (platform and "spotify" in platform.lower()) or ("spotify" in media_title.lower()):
        search_platform = "spotify"
        media_title = media_title.lower().replace("on spotify", "").strip()
    print(f"Attempting to play '{media_title}' on {search_platform}...")
    if search_platform == "spotify":
        search_url = f"https://open.spotify.com/search/{urllib.parse.quote(media_title)}"
    else:
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(media_title)}"
    webbrowser.open(search_url)

def control_media(control_command: str):
    """Controls media playback using keyboard media keys (macOS only)."""
    os_name = platform.system()
    if os_name != "Darwin":
        print(f"Media control is only supported on macOS for now.")
        return
    key_codes = {"play": "16", "pause": "16", "next": "19", "previous": "20"}
    key_code = key_codes.get(control_command.lower())
    if key_code:
        try:
            command = f'osascript -e \'tell application "System Events" to key code {key_code}\''
            subprocess.run(command, shell=True, check=True)
            print(f"Executed media command: {control_command}")
        except Exception as e:
            print(f"Error executing media command '{control_command}': {e}")
    else:
        print(f"Unknown media control command: {control_command}")

# --- Main Executor Router ---
def execute_action(intent: str, slots: dict):
    """Routes the predicted intent to the correct execution function."""
    print(f"--- Executing action for intent: {intent} with slots: {slots} ---")
    try:
        if intent == "os.app.open":
            open_app(slots.get("app_name", ""))
        elif intent == "os.app.close":
            close_app(slots.get("app_name", ""))
        elif intent == "browser.open":
            open_url(slots.get("url", ""))
        elif intent == "browser.search":
            search_web(slots.get("query", ""))
        elif intent == "productivity.note.create":
            create_note(slots.get("content", ""))
        elif intent == "media.play":
            title_keys = ["media_title", "song_title", "song_name", "song", "track", "title", "video_title", "query"]
            platform_keys = ["platform", "source", "service", "site", "app"]
            media_title = ""
            for k in title_keys:
                v = slots.get(k)
                if isinstance(v, str) and v.strip():
                    media_title = v.strip()
                    break
            platform = ""
            for k in platform_keys:
                v = slots.get(k)
                if isinstance(v, str) and v.strip():
                    platform = v.strip()
                    break
            lt = media_title.lower()
            if not platform and " on youtube" in lt:
                platform = "YouTube"
                media_title = media_title[:lt.rfind(" on youtube")].strip()
            if not platform and " on spotify" in lt:
                platform = "Spotify"
                media_title = media_title[:lt.rfind(" on spotify")].strip()
            if not media_title:
                print("No media title extracted from slots; got:", slots)
                return
            play_media(media_title, platform)
        elif intent == "media.pause":
            control_media("pause")
        elif intent == "media.resume":
            control_media("play")
        elif intent == "media.stop":
            control_media("pause")
        elif intent == "media.next":
            control_media("next")
        elif intent == "media.previous":
            control_media("previous")
        else:
            print(f"Execution for intent '{intent}' is not implemented yet.")
    except Exception as e:
        print(f"❌ Error executing {intent}: {e}")
        print("💡 Try being more specific or check if the app/service is available")
