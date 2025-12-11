import os
from dotenv import load_dotenv
from typing import Optional

# --- Configuration and Setup ---

# Load environment variables from a .env file
load_dotenv()

# Get the API Key and ensure it is present
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing GEMINI_API_KEY in .env file. Please create one.")

# Import Gemini SDK components
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "The Google GenAI SDK is not installed. "
        "Install with: pip install --upgrade google-genai python-dotenv"
    )

# Initialize the Gemini client and set the model
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"


# --- Helper Function for Robust Text Extraction ---

def extract_text(response: types.GenerateContentResponse) -> str:
    """
    Robustly extracts the text content from a Gemini API response object,
    handling various response structures gracefully.
    """
    # 1. Direct access via the 'text' attribute (most common success case)
    if hasattr(response, "text") and response.text:
        return response.text.strip()

    # 2. Extracting from candidates/content/parts (fallback for complex responses)
    try:
        if hasattr(response, "candidates"):
            cand = response.candidates[0]

            if hasattr(cand, "content"):
                content = cand.content
                if hasattr(content, "parts"):
                    out = []
                    for p in content.parts:
                        if hasattr(p, "text") and p.text:
                            out.append(p.text)
                        elif hasattr(p, "content") and p.content:
                            out.append(str(p.content))
                    if out:
                        return "\n".join(out).strip()

                if hasattr(content, "text") and content.text:
                    return content.text.strip()
    except Exception:
        # Ignore intermediate errors and proceed to final fallbacks
        pass

    # 3. Final fallback: convert the entire response object to string
    try:
        return str(response)
    except Exception:
        return "NO_TEXT_FOUND"


# --- Main Name Generation Function ---

def generate_creative_names(
    name_type: str,
    theme: str,
    count: int = 5,
    constraints: Optional[str] = None,
    temperature: float = 0.9,
) -> str:
    """
    Generates a list of creative names using the Gemini model with a strict format.

    Args:
        name_type: The category of names (e.g., 'company names', 'character names').
        theme: The subject or genre for the names (e.g., 'AI startups', 'Space Opera').
        count: The number of names to generate (clamped between 1 and 20).
        constraints: Specific requirements for the names (e.g., 'must start with Z').
        temperature: Creativity level (0.0 to 1.0).

    Returns:
        The formatted list of names as a string, or an error message.
    """
    # Ensure count is within a reasonable range for API efficiency
    count = max(1, min(count, 20))

    # Strict instruction to guide the model's tone and format
    system_instruction = (
        "You are a world-class naming expert. ALWAYS be concise. "
        "Each name + explanation must be UNDER 8 words total. "
        "Only output the numbered list in the exact format. "
        "Follow all user constraints softly but prioritize quality."
    )

    constraint_text = f"Constraints: {constraints}." if constraints else ""

    # The user prompt defining the task and required format
    prompt = (
        f"Generate exactly {count} creative {name_type} for the theme: \"{theme}\".\n"
        f"{constraint_text}\n\n"
        "Strict output format:\n"
        "1. NAME — Etymology/Meaning: ... | Why it works: ...\n"
        "2. NAME — ...\n"
        "Only the numbered list."
    )

    # Configuration for the API call
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=800, # A safe limit for a short list of names
    )

    print(f"Sending request to {MODEL_NAME}...")

    try:
        # Call the Gemini API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=cfg,
        )
        return extract_text(response)
    except Exception as e:
        return f"API Error: Failed to generate content: {e}"


# --- Script Execution ---

# --- Script Execution ---

def main():
    """Contains the main logic for user interaction and script execution."""
    print("\n=== Gemini API Creative Name Generator ===\n")
    print("This script connects to the Gemini model to brainstorm names.\n")

    # Gather required inputs from the user
    name_type = input("Enter the type of names (e.g., company names, character names): ")
    theme = input("Enter Theme/genre: ")
    count = 5  # Default, can be made user-configurable later

    constraints = input(
        "Enter all requirements (e.g., length, starting letter, style, banned letters): "
    ).strip()
    constraints = constraints if constraints else None

    if not name_type or not theme:
        print("\nType and Theme are required. Exiting.")
        return # Exit the function cleanly
    else:
        print("\nGenerating...\n")

        # Execute the main generation function
        result = generate_creative_names(
            name_type=name_type,
            theme=theme,
            count=count,
            constraints=constraints,
        )

        # Print the final result
        print(result)
        print("\n=== GENERATION COMPLETE ===\n")


if __name__ == "__main__":
    main() # Call the new main function to start execution
