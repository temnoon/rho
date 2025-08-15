"""
Un-Earthify System: Transform Earth-based narratives into alien contexts
while preserving their essential ρ-space meaning.

This system systematically replaces Earth-specific references with 
pronounceable, plausible alien alternatives, maintaining consistency
across multiple texts to build a coherent parallel reality.
"""

import re
import json
import logging
import random
from typing import Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict

from config import config

logger = logging.getLogger(__name__)

@dataclass
class AlienMapping:
    """Maps Earth concept to alien equivalent"""
    earth_term: str
    alien_term: str
    category: str  # "planet", "animal", "food", "technology", etc.
    context_examples: List[str]  # Examples of usage contexts
    phonetic_pattern: str  # Pattern used to generate the alien word
    usage_count: int = 0

class AlienWordGenerator:
    """Generates pronounceable alien words following linguistic patterns"""
    
    # Alien phoneme patterns for different categories (shorter names)
    PHONEME_PATTERNS = {
        "planet": ["Xar", "Kel", "Mor", "Vel", "Zyx"],
        "species": ["Vex", "Kri", "Zol", "Nyx", "Qex"],  
        "food": ["Syl", "Pyx", "Kex", "Lyx", "Nex"],
        "technology": ["Tek", "Vox", "Zar", "Nex", "Qux"],
        "place": ["Dal", "Kyr", "Mor", "Vel", "Zex"],
        "material": ["Tyx", "Nox", "Vex", "Kex", "Zyl"],
        "concept": ["Lux", "Vex", "Zyx", "Nex", "Qux"]
    }
    
    SUFFIXES = {
        "planet": ["on", "ex", "yx", "al", "ur"],
        "species": ["ar", "ex", "yx", "on", "ul"],
        "food": ["yx", "ex", "on", "ul", "ar"],
        "technology": ["ex", "yx", "on", "ar", "ul"],
        "place": ["ex", "on", "yx", "ur", "al"],
        "material": ["yx", "ex", "on", "ul", "ar"],
        "concept": ["ex", "yx", "on", "ar", "ul"]
    }
    
    def generate_alien_word(self, earth_term: str, category: str) -> str:
        """Generate a pronounceable alien equivalent for an Earth term"""
        
        # Choose phoneme pattern based on category
        phonemes = self.PHONEME_PATTERNS.get(category, self.PHONEME_PATTERNS["concept"])
        suffixes = self.SUFFIXES.get(category, self.SUFFIXES["concept"])
        
        # Generate base word (favor shorter names)
        base = random.choice(phonemes)
        
        # Only add complexity for very long earth terms
        if len(earth_term) > config.ALIEN_COMPLEXITY_THRESHOLD:
            base += random.choice(phonemes).lower()
        
        # Add appropriate suffix (shorter suffixes)
        suffix = random.choice(suffixes)
        alien_word = base + suffix
        
        # Ensure it's pronounceable and not too long
        alien_word = self._make_pronounceable(alien_word)
        
        # Cap at configured max length for readability
        if len(alien_word) > config.ALIEN_WORD_MAX_LENGTH:
            alien_word = alien_word[:config.ALIEN_WORD_MAX_LENGTH]
        
        return alien_word
    
    def _make_pronounceable(self, word: str) -> str:
        """Ensure the generated word is pronounceable"""
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        
        result = []
        prev_was_consonant = False
        
        for i, char in enumerate(word.lower()):
            if char in consonants:
                if prev_was_consonant and i > 0:
                    # Insert vowel between consonants
                    result.append(random.choice(vowels))
                result.append(char)
                prev_was_consonant = True
            else:
                result.append(char)
                prev_was_consonant = False
        
        return ''.join(result).capitalize()

class EarthTermCategorizer:
    """Categorizes Earth terms for appropriate alien replacement"""
    
    # Only Earth-specific terms that should be replaced
    EARTH_SPECIFIC_TERMS = {
        "planets": ["earth", "mars", "venus", "jupiter", "saturn", "mercury", "neptune", "uranus", "pluto"],
        "animals": ["dog", "cat", "horse", "cow", "pig", "chicken", "elephant", "lion", "tiger", "bear"],
        "foods": ["coffee", "tea", "wine", "beer", "bread", "rice", "wheat", "corn", "apple", "orange", "banana"],
        "places": ["america", "europe", "asia", "africa", "australia", "usa", "canada", "russia", "china", "japan", 
                  "california", "new york", "london", "paris", "tokyo", "beijing", "stanford", "harvard", "mit"],
        "technologies": ["internet", "smartphone", "television", "radio", "car", "airplane", "train"],
        "materials": ["steel", "iron", "gold", "silver", "copper", "aluminum", "plastic", "wood"],
        "religions": ["christianity", "islam", "judaism", "buddhism", "hinduism", "taoism"],
        "religious_terms": ["buddha", "christ", "allah", "vishnu", "dao", "sutra", "bible", "quran", "torah"],
        "cultural_concepts": ["democracy", "capitalism", "socialism", "communism", "feudalism"],
        "currencies": ["dollar", "euro", "yen", "pound", "bitcoin"],
        "organizations": ["nasa", "fbi", "cia", "un", "nato", "who", "google", "microsoft", "apple"],
        "languages": ["english", "spanish", "french", "german", "chinese", "japanese", "russian"],
        "proper_names": ["yoneda", "einstein", "newton", "plato", "aristotle", "confucius"],
        "earth_specific_terms": ["catuskoti", "vimalakirti", "indra", "mahayana", "zen", "sanskrit"]
    }
    
    # Universal concepts that should NOT be replaced
    UNIVERSAL_CONCEPTS = {
        "physical", "consciousness", "experience", "understanding", "logic", "mathematics", "theory", 
        "philosophy", "science", "knowledge", "truth", "reality", "existence", "time", "space",
        "mind", "thought", "perception", "awareness", "intelligence", "reason", "intuition",
        "category", "structure", "system", "process", "method", "analysis", "synthesis",
        "objective", "subjective", "relative", "absolute", "infinite", "finite", "necessary",
        "possible", "actual", "potential", "cause", "effect", "relation", "connection",
        "pattern", "form", "content", "meaning", "purpose", "value", "quality", "quantity",
        "unity", "diversity", "similarity", "difference", "identity", "change", "stability",
        "order", "chaos", "simple", "complex", "whole", "part", "general", "particular",
        "abstract", "concrete", "universal", "individual", "essence", "appearance"
    }
    
    def categorize_term(self, term: str) -> Optional[str]:
        """Determine if an Earth term should be replaced and what category it belongs to"""
        term_lower = term.lower()
        
        # First check if it's a universal concept that should NOT be replaced
        if term_lower in self.UNIVERSAL_CONCEPTS:
            return None  # Don't replace universal concepts
        
        # Check if it's in our Earth-specific terms that should be replaced
        for category, terms in self.EARTH_SPECIFIC_TERMS.items():
            if term_lower in terms:
                return category
        
        # Pattern-based categorization for Earth-specific terms only
        if term_lower.endswith(('ism', 'ity')) and term_lower not in self.UNIVERSAL_CONCEPTS:
            return "cultural_concepts"
        elif term_lower.endswith(('istan', 'land', 'ia', 'ica')):
            return "places"
        elif re.match(r'^[A-Z][a-z]+$', term) and len(term) > 4:
            # Proper nouns might be places or names, but be conservative
            return "proper_names"
        
        return None  # Default: don't replace

class UnEarthifyEngine:
    """Main engine for systematic un-earthification"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        
        self.generator = AlienWordGenerator()
        self.categorizer = EarthTermCategorizer()
        self.mappings: Dict[str, AlienMapping] = {}
        self.alien_dictionary: Dict[str, str] = {}  # alien -> earth reverse lookup
        
        self._load_mappings()
    
    def _load_mappings(self):
        """Load existing alien mappings from disk"""
        mapping_file = config.get_alien_mappings_path()
        
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    data = json.load(f)
                    
                for item in data.get("mappings", []):
                    mapping = AlienMapping(**item)
                    self.mappings[mapping.earth_term] = mapping
                    self.alien_dictionary[mapping.alien_term] = mapping.earth_term
                    
                logger.info(f"Loaded {len(self.mappings)} alien mappings")
            except Exception as e:
                logger.error(f"Failed to load alien mappings: {e}")
    
    def _save_mappings(self):
        """Save alien mappings to disk"""
        mapping_file = config.get_alien_mappings_path()
        
        try:
            data = {
                "mappings": [asdict(mapping) for mapping in self.mappings.values()],
                "total_mappings": len(self.mappings)
            }
            
            with open(mapping_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.mappings)} alien mappings")
        except Exception as e:
            logger.error(f"Failed to save alien mappings: {e}")
    
    def identify_earth_terms(self, text: str) -> List[str]:
        """Identify Earth-specific terms in text that should be replaced"""
        earth_terms = set()
        
        # Direct lookup in Earth-specific terms only
        for category, terms in self.categorizer.EARTH_SPECIFIC_TERMS.items():
            for term in terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(text):
                    earth_terms.add(term.lower())
        
        # Pattern-based identification for proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        for noun in proper_nouns:
            category = self.categorizer.categorize_term(noun)
            if category:  # Only add if categorize_term returns a category (not None)
                earth_terms.add(noun.lower())
        
        return list(earth_terms)
    
    def get_or_create_mapping(self, earth_term: str, context: str) -> AlienMapping:
        """Get existing mapping or create new one for Earth term"""
        
        if earth_term in self.mappings:
            # Update usage count and context
            mapping = self.mappings[earth_term]
            mapping.usage_count += 1
            if context not in mapping.context_examples:
                mapping.context_examples.append(context[:config.CONTEXT_WINDOW_SIZE])  # Truncate long contexts
        else:
            # Create new mapping
            category = self.categorizer.categorize_term(earth_term)
            alien_term = self.generator.generate_alien_word(earth_term, category)
            
            # Ensure unique alien term
            while alien_term.lower() in self.alien_dictionary:
                alien_term = self.generator.generate_alien_word(earth_term, category)
            
            mapping = AlienMapping(
                earth_term=earth_term,
                alien_term=alien_term,
                category=category,
                context_examples=[context[:config.CONTEXT_WINDOW_SIZE]],
                phonetic_pattern=f"{category}_pattern",
                usage_count=1
            )
            
            self.mappings[earth_term] = mapping
            self.alien_dictionary[alien_term.lower()] = earth_term
            
            logger.info(f"Created mapping: {earth_term} -> {alien_term} ({category})")
        
        return mapping
    
    def un_earthify_text(self, text: str, preserve_essence: bool = True) -> Dict:
        """Transform Earth-based text into alien equivalent"""
        
        # Identify Earth terms
        earth_terms = self.identify_earth_terms(text)
        
        if not earth_terms:
            return {
                "original_text": text,
                "un_earthified_text": text,
                "transformations": [],
                "rho_preservation": True
            }
        
        # Create mappings and transform text
        transformed_text = text
        transformations = []
        
        for earth_term in earth_terms:
            # Get context around the term
            pattern = re.compile(r'.{0,50}\b' + re.escape(earth_term) + r'\b.{0,50}', re.IGNORECASE)
            match = pattern.search(text)
            context = match.group(0) if match else text[:100]
            
            # Get or create mapping
            mapping = self.get_or_create_mapping(earth_term, context)
            
            # Replace in text (case-preserving)
            def replace_func(match):
                original = match.group(0)
                if original[0].isupper():
                    return mapping.alien_term.capitalize()
                else:
                    return mapping.alien_term.lower()
            
            pattern = re.compile(r'\b' + re.escape(earth_term) + r'\b', re.IGNORECASE)
            transformed_text = pattern.sub(replace_func, transformed_text)
            
            transformations.append({
                "earth_term": earth_term,
                "alien_term": mapping.alien_term,
                "category": mapping.category,
                "context": context
            })
        
        # Save updated mappings
        self._save_mappings()
        
        return {
            "original_text": text,
            "un_earthified_text": transformed_text,
            "transformations": transformations,
            "earth_terms_found": len(earth_terms),
            "rho_preservation": preserve_essence  # In real implementation, would verify this
        }
    
    def get_alien_dictionary(self) -> Dict[str, AlienMapping]:
        """Get complete alien dictionary for reference"""
        return self.mappings.copy()
    
    def reverse_mapping(self, alien_text: str) -> str:
        """Convert alien text back to Earth terms (for debugging)"""
        earth_text = alien_text
        
        for alien_term, earth_term in self.alien_dictionary.items():
            pattern = re.compile(r'\b' + re.escape(alien_term) + r'\b', re.IGNORECASE)
            earth_text = pattern.sub(earth_term, earth_text)
        
        return earth_text

# Global instance
un_earthify_engine = UnEarthifyEngine()

def un_earthify_text(text: str) -> Dict:
    """Public interface for un-earthifying text"""
    return un_earthify_engine.un_earthify_text(text)

def get_alien_dictionary() -> Dict[str, AlienMapping]:
    """Get the complete alien dictionary"""
    return un_earthify_engine.get_alien_dictionary()

if __name__ == "__main__":
    # Test the un-earthify system
    test_text = """
    The American astronaut landed on Mars after a long journey from Earth. 
    He missed his coffee and wished NASA had sent better food. The red planet 
    looked nothing like the green fields of Ohio where he grew up.
    """
    
    result = un_earthify_text(test_text)
    
    print("Original:", result["original_text"])
    print("\nUn-Earthified:", result["un_earthified_text"])
    print(f"\nTransformations ({result['earth_terms_found']}):")
    for t in result["transformations"]:
        print(f"  {t['earth_term']} -> {t['alien_term']} ({t['category']})")