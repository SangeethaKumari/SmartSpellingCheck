## Smart Spelling Check – Agentic Grammar & Spell Checker

SmartSpellingCheck is a **Streamlit web app** that uses **Google Gemini** to intelligently fix user text.  
Instead of only correcting typos, it can:

- **Reconstruct famous phrases** when the words are wrong but spelled correctly (semantic reconstruction).
- **Fix grammar and sentence structure** (word order, tense, subject–verb agreement, etc.).
- **Perform simple spell checking** for normal typos.

This project was created for the Kaggle **Agents Intensive Capstone Project** and is packaged as a modern `uv`-managed Python app.

---

## Project Structure & Code Overview

- **`pyproject.toml`**
  - Declares the project as `smartspellingcheck`, Python `>=3.12`.
  - Depends on:
    - **`google-generativeai`**: official client for the Gemini API.
    - **`generativeai`** and **`svlearn-bootcamp`**: helper libraries used in the SupportVectors bootcamp environment.

- **`src/smart_spelling_check/spell_check.py`**
  - This file contains **all the main logic and the Streamlit UI**.

### 1. `GeminiLLM` – Gemini API wrapper

- **What it does**
  - Wraps Gemini configuration and calls into a small, reusable client.
- **Key parts**
  - On initialization, it configures Gemini with the given API key:
    - `genai.configure(api_key=api_key, transport="rest")`
    - Creates `genai.GenerativeModel("gemini-2.5-flash")`.
  - `call(prompt, use_json=True)`:
    - Sends the prompt to Gemini.
    - If `use_json=True`, it asks Gemini to return **JSON** (`response_mime_type="application/json"`).
    - Returns `response.text` which is then parsed with `json.loads(...)` by the tools.

### 2. `GrammarTools` – Agent tools over the LLM

`GrammarTools` is a collection of **static methods** that all:

- Build a **very explicit prompt** for Gemini.
- Ask for a **JSON response** with a strict schema.
- `json.loads(...)` that response into Python dictionaries.

The tools:

- **`analyze_text(text, llm)`**
  - Asks Gemini to classify what kind of issues the text has:
    - `needs_reconstruction`: scrambled or wrong words in a known phrase.
    - `needs_grammar_fix`: grammar / sentence-structure issues.
    - `needs_spell_check`: simple spelling errors.
    - `severity`: `"high" | "medium" | "low"`.
    - `reasoning`: natural-language explanation.

- **`fix_grammar(text, llm)`**
  - (Currently not wired into the main agent decision, but fully defined.)
  - Prompt asks Gemini to:
    - Fix word order, prepositions, tense, subject–verb agreement, structure.
    - Return JSON with:
      - `has_errors`
      - `original`
      - `corrected`
      - `errors`: each with `error_type`, `wrong`, `correct`, `explanation`.

- **`semantic_reconstruction(text, llm)`**
  - Designed for **famous quotes / proverbs** that are scrambled or use wrong words.
  - Prompt:
    - Explains that words might be spelled correctly but be **wrong words**.
    - Gives examples: *“pen is mightier than the pencil” → “pen is mightier than the sword”*.
    - Asks Gemini to:
      - Identify which known phrase the user likely meant.
      - Reconstruct the **complete, correct phrase**.
      - Return JSON with:
        - `reconstructed`
        - `confidence` (0–100)
        - `reasoning`
        - `words_changed` (even if spelling was correct but semantics were wrong).

- **`detect_errors(text, llm)`**
  - Finds spelling errors and returns:
    - `has_errors`
    - `errors`: each with `error_text`, `correct_spelling`, `explanation`.

- **`fix_errors(text, errors, llm)`**
  - Takes the original text and the list of errors.
  - Asks Gemini to return:
    - `corrected_text`
    - `changes` (description of exactly what was changed).

### 3. `SmartGrammarAgent` – The agentic decision-maker

`SmartGrammarAgent` is where the **“agentic” behavior** lives.  
It decides **which toolchain to use** based on an initial analysis step, and logs all its actions.

- **Initialization**
  - Stores the `GeminiLLM` instance.
  - Sets up a `GrammarTools` instance.
  - Maintains a `steps` list to log reasoning.

- **`log_step(step, content)`**
  - Appends a dict `{"step": step, "content": content}` to `steps`.
  - Used to display a **step-by-step trace** in the Streamlit UI.

- **`run(text)` – Main agent loop**
  1. **OBSERVE**  
     - Logs the input text.
  2. **ANALYZE**  
     - Calls `GrammarTools.analyze_text(...)`.
     - Logs the raw JSON of the analysis.
  3. **DECIDE**
     - If `analysis["needs_reconstruction"]` is `True`:
       - Takes the **semantic reconstruction path**.
     - Otherwise:
       - Currently defaults to the **spell-checking path**.
       - (The grammar-correction path is defined in tools, and the UI has a slot for it, but the agent currently chooses between reconstruction vs. spell-check.)
  4. **Semantic reconstruction path**
     - Calls `semantic_reconstruction(...)` and logs results.
     - Then builds a **verification prompt** to double-check that the reconstructed text is:
       - A real, known phrase.
       - Correct and complete.
     - If verification says it is not correct, it uses `correct_phrase` from the verifier.
     - Then it runs `detect_errors(...)` and `fix_errors(...)` on the reconstructed phrase to catch any leftover typos.
     - Returns a dict including:
       - `original`, `method="semantic_reconstruction"`, `reconstructed`, `confidence`, `reasoning`, `final_text`, and `steps`.
  5. **Spell-checking path**
     - Calls `detect_errors(...)` on the original text.
     - If there are errors, calls `fix_errors(...)` to get a corrected version.
     - Returns:
       - `original`, `method="spell_check"`, `has_errors`, `errors`, `corrected`, `changes`, `final_text`, and `steps`.

The **agent’s internal reasoning** is fully exposed to the UI so users can see *why* it chose semantic reconstruction vs. simple spelling.

### 4. Streamlit UI (`main()` function)

The bottom of `spell_check.py` defines a full **Streamlit user interface**.

- **Page setup**
  - `st.set_page_config(page_title="Smart Agentic Spell Checker", layout="wide")`.
  - Title and description referencing the Kaggle hackathon and what the agent does.

- **Sidebar**
  - Accepts a **Gemini API key**:
    - Input box (`st.text_input(..., type="password")`).
    - If empty, tries `os.environ["GEMINI_API_KEY"]`.
  - Shows instructions on where to get an API key and what the agent can do.
  - If no API key is available, the app **stops** and shows a warning.

- **Main layout**
  - Two columns:
    - **Left (`col1`) – Input**
      - A **select box** of examples:
        - Phrases with grammar errors.
        - Famous quotes scrambled or with wrong words.
        - Simple spelling mistakes.
      - A text area for custom input or a pre-filled example.
      - A big **“Run Smart Agent”** button.
    - **Right (`col2`) – Agent Process & Results**
      - When the button is pressed:
        - Creates a `SmartGrammarAgent` instance.
        - Calls `agent.run(text)` in a `st.spinner("Agent analyzing...")`.
      - Shows:
        - A **list of agent steps** with expanders (OBSERVE, ANALYZE, DECIDE, etc.).
        - Below that, a section depending on `method`:
          - `semantic_reconstruction`:
            - Shows original, reconstructed text, confidence, reasoning, and a final result box.
          - `grammar_correction`:
            - (Reserved for future; UI ready to display grammar error details and corrected text.)
          - `spell_check`:
            - Lists each spelling error (wrong vs. correct) and the final corrected text.

- **Entry point**
  - At the bottom:
    - `if __name__ == "__main__": main()`
  - This allows running the app directly as a script (or via Streamlit).

---

## Running the Project with `uv`

This project is configured to be run using [`uv`](https://github.com/astral-sh/uv), a fast Python package and environment manager.

### 1. Prerequisites

- **Python**: 3.12 or later (as per `pyproject.toml`).
- **uv** installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, ensure `uv` is on your `PATH` (open a new terminal if needed).

### 2. Clone the repository

```bash
git clone <REPO_URL> SmartSpellingCheck
cd SmartSpellingCheck
```

Replace `<REPO_URL>` with the GitHub HTTPS or SSH URL.

### 3. Create and sync the environment

Install all dependencies into a `uv`-managed virtual environment:

```bash
uv sync
```

This will:

- Create an isolated environment for the project.
- Install the packages from `pyproject.toml` and `uv.lock`.

### 4. Provide your Gemini API key

You have **two options**:

- **Option A – Environment variable (recommended)**:

  ```bash
  export GEMINI_API_KEY="YOUR_API_KEY_HERE"
  ```

- **Option B – Enter in UI**:
  - Leave `GEMINI_API_KEY` unset.
  - When you open the app, paste your API key into the sidebar input.

You can get an API key from **Google AI Studio** (`https://makersuite.google.com/app/apikey`).

### 5. Run the Streamlit app using `uv`

Use `uv run` so that it automatically uses the project’s environment:

```bash
uv run streamlit run src/smart_spelling_check/spell_check.py
```

What this does:

- `uv run` launches the command inside the project’s environment.
- `streamlit run ...` starts the web app defined in `spell_check.py`.
- Streamlit will print a local URL (usually `http://localhost:8501`) – open it in your browser.

### 6. (Optional) Run as a Python module

Since `smart_spelling_check` is a Python package, you can also run the file as a module:

```bash
uv run python -m smart_spelling_check.spell_check
```

This will execute the same `main()` function and launch the Streamlit UI.

---

## Typical Workflow for Someone Cloning from GitHub

1. **Install uv** (once on your machine).
2. **Clone the repo**:
   - `git clone <REPO_URL> SmartSpellingCheck`
   - `cd SmartSpellingCheck`
3. **Sync the environment**:
   - `uv sync`
4. **Set your Gemini API key**:
   - `export GEMINI_API_KEY="YOUR_API_KEY_HERE"`  
     or paste the key into the sidebar when the app starts.
5. **Run the app**:
   - `uv run streamlit run src/smart_spelling_check/spell_check.py`
6. **Use the UI**:
   - Pick an example or type your own text.
   - Click **“Run Smart Agent”**.
   - Inspect the agent steps and the final corrected / reconstructed text.

Following these steps, anyone can **clone the project from GitHub, install dependencies with `uv`, and run the intelligent spell/grammar checker locally**.

