"""
Agentic Spell Checker with Semantic Reconstruction using Gemini API
Kaggle Hackathon: https://www.kaggle.com/competitions/agents-intensive-capstone-project
"""

import json
import streamlit as st
import google.generativeai as genai
import os

# ============================================================================
# Gemini Client
# ============================================================================

class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key, transport='rest')
        self.model = genai.GenerativeModel(model)
    
    def call(self, prompt: str, use_json: bool = True) -> str:
        if use_json:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
        else:
            response = self.model.generate_content(prompt)
        return response.text


# ============================================================================
# Agent Tools
# ============================================================================

class GrammarTools:
    
    @staticmethod
    def analyze_text(text: str, llm: GeminiLLM) -> dict:
        """Analyze if text needs semantic reconstruction, grammar fix, or just spell checking"""
        prompt = f"""Analyze this text: "{text}"

Determine what type of errors exist:
1. Wrong words in famous phrase? (e.g., "pen is mightier than pencil" vs "sword")
2. Grammar errors? (e.g., "correct me my speaking" - wrong sentence structure)
3. Scrambled/garbled with misspellings?
4. Just simple spelling errors?

Provide JSON with:
- needs_reconstruction: true if wrong words/scrambled/matches famous phrase incorrectly
- needs_grammar_fix: true if sentence structure/grammar is wrong
- needs_spell_check: true if just spelling errors
- severity: "high" (reconstruction), "medium" (grammar), "low" (spelling)
- reasoning: explain what's wrong"""
        
        response = llm.call(prompt)
        return json.loads(response)
    
    @staticmethod
    def fix_grammar(text: str, llm: GeminiLLM) -> dict:
        """Fix grammatical errors in sentence structure"""
        prompt = f"""Fix the grammar in this sentence: "{text}"

Look for:
- Wrong word order
- Missing/extra prepositions (to, for, with, at, in, etc.)
- Subject-verb agreement errors
- Incorrect tense
- Wrong sentence structure

Provide JSON with:
- has_errors: boolean
- original: "{text}"
- corrected: the grammatically correct sentence
- errors: array of grammar errors found with:
  - error_type: "word_order", "preposition", "subject_verb", "tense", "structure"
  - wrong: the incorrect part
  - correct: the correct part
  - explanation: why it's wrong"""
        
        response = llm.call(prompt)
        return json.loads(response)
    
    @staticmethod
    def semantic_reconstruction(text: str, llm: GeminiLLM) -> dict:
        """Reconstruct scrambled or garbled text to find intended meaning"""
        prompt = f"""Reconstruct this text that may have wrong words or structure: "{text}"

CRITICAL: The words might be spelled correctly but be WRONG words!
Example: "The pen is mightier than the pencil" - "pencil" is spelled right but WRONG word!
Correct: "The pen is mightier than the sword" (famous quote)

STEP 1 - Analyze the text:
- Are the words spelled correctly?
- Does the sentence make LOGICAL sense?
- Does it match a known famous phrase/quote/proverb?

STEP 2 - Match to famous phrases:
- What famous phrase has similar structure?
- What is the ACTUAL famous quote/saying?
- Don't just fix spelling - fix WRONG WORDS too!

STEP 3 - Reconstruct the EXACT original famous phrase

PATTERN EXAMPLES (for reference):
- "to X is Y, to Z is W" → check famous quotes
- "a X in time saves Y" → check proverbs
- "the X is mightier than the Y" → check historical quotes
- "all that X is not Y" → check idioms

Common famous phrases to consider:
- "The pen is mightier than the sword" (Edward Bulwer-Lytton)
- "To err is human, to forgive is divine" (Alexander Pope)
- "Actions speak louder than words"
- "A stitch in time saves nine"
- "Don't count your chickens before they hatch"
- "All that glitters is not gold"
- "The early bird catches the worm"

Provide JSON with:
- original: "{text}"
- reconstructed: the complete correct famous phrase (fix WRONG WORDS, not just spelling!)
- confidence: 0-100
- reasoning: which famous phrase this is, what was wrong (spelling OR wrong words)
- words_changed: list of words that were wrong (even if spelled correctly)"""
        
        response = llm.call(prompt)
        return json.loads(response)
    
    @staticmethod
    def detect_errors(text: str, llm: GeminiLLM) -> dict:
        """Detect spelling errors"""
        prompt = f"""Find all spelling errors in: "{text}"

Provide JSON with:
- has_errors: boolean
- errors: array with error_text, correct_spelling, explanation"""
        
        response = llm.call(prompt)
        return json.loads(response)
    
    @staticmethod
    def fix_errors(text: str, errors: list, llm: GeminiLLM) -> dict:
        """Fix spelling errors"""
        prompt = f"""Fix these spelling errors in: "{text}"

Errors: {json.dumps(errors)}

Provide JSON with:
- corrected_text: the fixed text
- changes: array describing what was changed"""
        
        response = llm.call(prompt)
        return json.loads(response)


# ============================================================================
# Agentic System
# ============================================================================

class SmartGrammarAgent:
    """Agent that decides: reconstruction, grammar fix, or spell check"""
    
    def __init__(self, llm: GeminiLLM):
        self.llm = llm
        self.tools = GrammarTools()
        self.steps = []
    
    def log_step(self, step: str, content: str):
        """Log agent steps"""
        self.steps.append({"step": step, "content": content})
    
    def run(self, text: str) -> dict:
        """Run the intelligent agent"""
        
        # Step 1: Observe
        self.log_step("OBSERVE", f"Input text: '{text}'")
        
        # Step 2: Analyze - decide what type of problem this is
        self.log_step("ANALYZE", "Determining if text needs reconstruction or just spell checking...")
        analysis = self.tools.analyze_text(text, self.llm)
        self.log_step("ANALYSIS_RESULT", json.dumps(analysis, indent=2))
        
        # Step 3: Decide and execute based on analysis
        if analysis.get('needs_reconstruction'):
            # Path A: Semantic Reconstruction
            self.log_step("DECIDE", "Text is scrambled/garbled - using SEMANTIC RECONSTRUCTION")
            
            self.log_step("RECONSTRUCT", "Figuring out the intended meaning...")
            reconstruction = self.tools.semantic_reconstruction(text, self.llm)
            self.log_step("RECONSTRUCT_RESULT", json.dumps(reconstruction, indent=2))
            
            # Verify the reconstruction is a real famous phrase
            reconstructed_text = reconstruction.get('reconstructed', text)
            
            # Double-check if this is the correct and complete phrase
            verify_prompt = f"""Verify this reconstruction:

Original scrambled: "{text}"
Reconstructed as: "{reconstructed_text}"

Is this reconstruction correct and complete?
- Is it a real famous phrase/quote/saying?
- Is it the full and accurate version?
- Does it match common knowledge?

Provide JSON with:
- is_correct: true if accurate, false if needs correction
- correct_phrase: the actual complete phrase if different (or same if correct)
- source: origin (e.g., "proverb", "Alexander Pope", "folk saying")"""
            
            verification = json.loads(self.llm.call(verify_prompt))
            
            if not verification.get('is_correct'):
                reconstructed_text = verification.get('correct_phrase', reconstructed_text)
                self.log_step("CORRECTION", f"Corrected reconstruction to: {reconstructed_text}")
            else:
                self.log_step("VERIFIED", f"Reconstruction verified as correct: {reconstructed_text}")
            
            # After reconstruction, check for any remaining errors
            reconstructed_text = reconstruction.get('reconstructed', text)
            self.log_step("VERIFY", f"Checking reconstructed text for remaining errors...")
            detection = self.tools.detect_errors(reconstructed_text, self.llm)
            
            if detection.get('has_errors'):
                self.log_step("FIX", "Fixing remaining spelling errors...")
                fixes = self.tools.fix_errors(reconstructed_text, detection.get('errors', []), self.llm)
                final_text = fixes.get('corrected_text', reconstructed_text)
            else:
                final_text = reconstructed_text
            
            self.log_step("COMPLETE", f"Final result: '{final_text}'")
            
            return {
                "original": text,
                "method": "semantic_reconstruction",
                "reconstructed": reconstruction.get('reconstructed', text),
                "confidence": reconstruction.get('confidence', 0),
                "reasoning": reconstruction.get('reasoning', ''),
                "final_text": final_text,
                "steps": self.steps
            }
        
        else:
            # Path C: Simple Spell Checking
            self.log_step("DECIDE", "Text just needs SPELL CHECKING (simple errors)")
            
            self.log_step("DETECT", "Looking for spelling errors...")
            detection = self.tools.detect_errors(text, self.llm)
            self.log_step("DETECT_RESULT", json.dumps(detection, indent=2))
            
            if detection.get('has_errors'):
                self.log_step("FIX", "Fixing spelling errors...")
                fixes = self.tools.fix_errors(text, detection.get('errors', []), self.llm)
                self.log_step("FIX_RESULT", json.dumps(fixes, indent=2))
                
                return {
                    "original": text,
                    "method": "spell_check",
                    "has_errors": True,
                    "errors": detection.get('errors', []),
                    "corrected": fixes.get('corrected_text', text),
                    "changes": fixes.get('changes', []),
                    "final_text": fixes.get('corrected_text', text),
                    "steps": self.steps
                }
            else:
                self.log_step("COMPLETE", "No errors found!")
                return {
                    "original": text,
                    "method": "spell_check",
                    "has_errors": False,
                    "errors": [],
                    "final_text": text,
                    "steps": self.steps
                }


# ============================================================================
# Streamlit Interface
# ============================================================================

def main():
    st.set_page_config(page_title="Smart Agentic Spell Checker", layout="wide")
    
    st.title("🤖 Smart Agentic Grammar & Spell Checker")
    st.markdown("""
    **Intelligent agent that decides**: Semantic Reconstruction → Grammar Fix → Spell Check
    
    **Kaggle Hackathon**: [5-Day Agentic Framework](https://www.kaggle.com/competitions/agents-intensive-capstone-project)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza..."
        )
        
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        
        st.info("""
        **Get API Key:**
        [Google AI Studio](https://makersuite.google.com/app/apikey)
        
        **Model:** Gemini 2.5 Flash
        
        **Agent can handle:**
        - 🧠 Semantic reconstruction (scrambled famous phrases)
        - 📝 Grammar correction (sentence structure)
        - ✏️ Spell checking (simple typos)
        """)
    
    if not api_key:
        st.warning("Please enter your API key in the sidebar")
        st.stop()
    
    # Initialize
    llm = GeminiLLM(api_key=api_key)
    
    # Main interface
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.header("📝 Input")
        
        example = st.selectbox(
            "Try an example:",
            [
                "Custom",
                "correct me my speeking",
                "She don't likes apples",
                "He go to school yesterday",
                "The pen is mightier than the pencil",
                "to err is huma to forgiv is hman",
                "a stich in time saves nien",
                "I havv a speling eror"
            ]
        )
        
        if example == "Custom":
            text = st.text_area("Enter text:", height=200, placeholder="Type scrambled or misspelled text...")
        else:
            text = st.text_area("Enter text:", value=example, height=200)
        
        run_button = st.button("🧠 Run Smart Agent", type="primary", use_container_width=True)
    
    with col2:
        st.header("🔄 Agent Process")
        
        if run_button and text:
            agent = SmartGrammarAgent(llm)
            
            with st.spinner("Agent analyzing..."):
                result = agent.run(text)
            
            # Show agent steps
            st.subheader("🤖 Agent Decision Process:")
            for i, step in enumerate(result['steps'], 1):
                with st.expander(f"**Step {i}: {step['step']}**", expanded=(i <= 3)):
                    st.text(step['content'])
            
            st.markdown("---")
            
            # Show results based on method used
            method = result.get('method')
            
            if method == 'semantic_reconstruction':
                st.subheader("🧠 Semantic Reconstruction Used")
                
                with st.container(border=True):
                    st.markdown("**Original (Scrambled):**")
                    st.code(result['original'])
                    
                    st.markdown("**Reconstructed:**")
                    st.success(result['reconstructed'])
                    
                    st.markdown(f"**Confidence:** {result.get('confidence', 0)}%")
                    st.markdown(f"**Reasoning:** {result.get('reasoning', '')}")
                
                st.subheader("✅ Final Text:")
                st.info(result['final_text'])
            
            elif method == 'grammar_correction':
                st.subheader("📝 Grammar Correction Used")
                
                grammar_errors = result.get('grammar_errors', [])
                if grammar_errors:
                    st.warning(f"Found {len(grammar_errors)} grammar error(s)")
                    
                    for i, error in enumerate(grammar_errors, 1):
                        with st.container(border=True):
                            st.markdown(f"**Grammar Error {i}: {error.get('error_type', 'error').upper()}**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Wrong:**")
                                st.code(error.get('wrong', ''))
                            with col_b:
                                st.markdown("**Correct:**")
                                st.code(error.get('correct', ''), language="diff")
                            st.caption(error.get('explanation', ''))
                
                st.subheader("✅ Corrected Text:")
                st.success(result['final_text'])
            
            elif method == 'spell_check':
                st.subheader("✏️ Spell Checking Used")
                
                if result['has_errors']:
                    st.warning(f"Found {len(result['errors'])} error(s)")
                    
                    for i, error in enumerate(result['errors'], 1):
                        with st.container(border=True):
                            st.markdown(f"**Error {i}:**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Wrong:**")
                                st.code(error.get('error_text', ''))
                            with col_b:
                                st.markdown("**Correct:**")
                                st.code(error.get('correct_spelling', ''), language="diff")
                            st.caption(error.get('explanation', ''))
                    
                    st.subheader("✅ Corrected Text:")
                    st.success(result['corrected'])
                else:
                    st.success("✨ No spelling errors found!")
                    st.info(result['original'])


if __name__ == "__main__":
    main()