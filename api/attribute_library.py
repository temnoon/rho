#!/usr/bin/env python3
"""
Comprehensive Attribute Library for Rho Transformations

This module provides extensive collections of attributes across different
categories (namespace, persona, style) with management and favorites functionality.
"""

import json
import os
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttributeDefinition:
    """Definition of a single attribute"""
    name: str
    category: str  # namespace, persona, style
    subcategory: str  # more specific grouping
    description: str
    positive_examples: List[str]
    negative_examples: List[str]
    strength_range: Tuple[float, float] = (-2.0, 2.0)
    default_value: float = 0.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class AttributeCollection:
    """A collection of related attributes"""
    name: str
    description: str
    attributes: List[AttributeDefinition]
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class AttributeLibrary:
    """Comprehensive library of writing attributes"""
    
    def __init__(self, data_dir: str = "rho/api/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.attributes = {}  # name -> AttributeDefinition
        self.collections = {}  # name -> AttributeCollection
        self.favorites = set()  # Set of favorite attribute names
        self.user_attributes = {}  # Custom user-created attributes
        
        self._initialize_default_attributes()
        self._load_user_data()
    
    def _initialize_default_attributes(self):
        """Initialize the comprehensive default attribute library"""
        
        # NAMESPACE ATTRIBUTES - Different domains/worlds/contexts
        namespace_attrs = [
            # Historical periods
            AttributeDefinition(
                name="ancient_classical",
                category="namespace", 
                subcategory="historical",
                description="Ancient Greek/Roman classical perspective",
                positive_examples=[
                    "In the agora of Athens, philosophers debated",
                    "The Senate of Rome decreed",
                    "As Homer sang of heroes",
                    "The oracle at Delphi proclaimed"
                ],
                negative_examples=[
                    "The smartphone buzzed with notifications",
                    "Cloud computing revolutionized data storage",
                    "The GPS navigation system calculated"
                ],
                tags=["historical", "classical", "ancient"]
            ),
            
            AttributeDefinition(
                name="medieval_feudal",
                category="namespace",
                subcategory="historical", 
                description="Medieval feudal society and mindset",
                positive_examples=[
                    "The lord of the manor commanded",
                    "By royal decree and divine right",
                    "The guild masters assembled",
                    "In the cathedral's shadow"
                ],
                negative_examples=[
                    "Democratic principles guided the decision",
                    "Scientific method revealed the truth",
                    "Individual rights were paramount"
                ],
                tags=["historical", "medieval", "feudal"]
            ),
            
            AttributeDefinition(
                name="renaissance_humanist",
                category="namespace",
                subcategory="historical",
                description="Renaissance humanism and artistic flowering",
                positive_examples=[
                    "The artists of Florence perfected",
                    "Human potential knows no bounds",
                    "Through reason and beauty we ascend",
                    "The patron commissioned a masterwork"
                ],
                negative_examples=[
                    "Divine mandate supersedes human will",
                    "Traditional ways must never change",
                    "Art serves only practical purposes"
                ],
                tags=["historical", "renaissance", "humanist", "artistic"]
            ),
            
            AttributeDefinition(
                name="enlightenment_rational",
                category="namespace",
                subcategory="historical",
                description="Enlightenment rationalism and progress",
                positive_examples=[
                    "Reason illuminates the path forward",
                    "Natural laws govern all phenomena", 
                    "Progress through scientific inquiry",
                    "The rights of man are self-evident"
                ],
                negative_examples=[
                    "Mystical forces beyond understanding",
                    "Ancient traditions hold all wisdom",
                    "Faith alone guides human action"
                ],
                tags=["historical", "enlightenment", "rational", "scientific"]
            ),
            
            AttributeDefinition(
                name="industrial_mechanical",
                category="namespace",
                subcategory="historical",
                description="Industrial age mechanical worldview",
                positive_examples=[
                    "The great machines of progress churned",
                    "Efficiency and production maximized",
                    "Steel and steam transformed the world",
                    "The factory system organized labor"
                ],
                negative_examples=[
                    "Handcrafted artisanal methods preserved",
                    "Natural rhythms guide human activity",
                    "Individual creativity over mass production"
                ],
                tags=["historical", "industrial", "mechanical", "progress"]
            ),
            
            # Futuristic/Sci-fi
            AttributeDefinition(
                name="cyberpunk_dystopian",
                category="namespace",
                subcategory="futuristic",
                description="Cyberpunk high-tech low-life aesthetic",
                positive_examples=[
                    "Neon reflections in puddles of corporate waste",
                    "Neural interfaces bypassed organic limitations",
                    "The megacorp's data fortress loomed",
                    "Street hackers jacked into the matrix"
                ],
                negative_examples=[
                    "Natural harmony guides all decisions",
                    "Government serves the people's needs",
                    "Technology enhances human connection"
                ],
                tags=["futuristic", "cyberpunk", "dystopian", "tech"]
            ),
            
            AttributeDefinition(
                name="space_exploration",
                category="namespace",
                subcategory="futuristic",
                description="Space-faring civilization perspective",
                positive_examples=[
                    "Among the stars, humanity found purpose",
                    "The colony ship reached the distant system",
                    "Terraforming proceeded according to schedule",
                    "Galactic trade routes connected worlds"
                ],
                negative_examples=[
                    "Earth-bound thinking limited imagination",
                    "Local concerns overshadowed cosmic vision",
                    "Planetary boundaries defined reality"
                ],
                tags=["futuristic", "space", "exploration", "cosmic"]
            ),
            
            AttributeDefinition(
                name="post_scarcity",
                category="namespace",
                subcategory="futuristic",
                description="Post-scarcity abundance society",
                positive_examples=[
                    "Material wants became obsolete",
                    "Creativity flourished without economic constraint",
                    "The replicators provided endless abundance",
                    "Purpose replaced survival as motivation"
                ],
                negative_examples=[
                    "Limited resources demanded careful allocation",
                    "Competition drove innovation and progress",
                    "Scarcity created value and meaning"
                ],
                tags=["futuristic", "post-scarcity", "abundance", "utopian"]
            ),
            
            # Fantasy/Mythological
            AttributeDefinition(
                name="high_fantasy",
                category="namespace",
                subcategory="fantasy",
                description="High fantasy magical realm perspective",
                positive_examples=[
                    "Ancient magics stirred in the ether",
                    "The dragon's wisdom echoed through ages",
                    "Mystical energies flowed through ley lines",
                    "The prophecy foretold great change"
                ],
                negative_examples=[
                    "Scientific explanation demystified the phenomenon",
                    "Rational analysis revealed the cause",
                    "Technology provided practical solutions"
                ],
                tags=["fantasy", "magical", "mystical", "mythological"]
            ),
            
            AttributeDefinition(
                name="urban_fantasy",
                category="namespace",
                subcategory="fantasy",
                description="Modern world with hidden magic",
                positive_examples=[
                    "The coffee shop concealed a portal",
                    "Magic hid behind mundane facades",
                    "Supernatural politics infiltrated city hall",
                    "The smartphone app was actually a spell"
                ],
                negative_examples=[
                    "Pure medieval fantasy setting",
                    "Completely technological solutions",
                    "No hidden supernatural elements"
                ],
                tags=["fantasy", "urban", "hidden", "contemporary"]
            ),
            
            # Academic/Professional
            AttributeDefinition(
                name="academic_scholarly",
                category="namespace",
                subcategory="professional",
                description="Academic scholarly discourse",
                positive_examples=[
                    "The research methodology employed",
                    "Peer review established validity",
                    "The literature review revealed",
                    "Further investigation is warranted"
                ],
                negative_examples=[
                    "My gut feeling tells me",
                    "Everyone knows that",
                    "It's obvious to anyone"
                ],
                tags=["academic", "scholarly", "research", "formal"]
            ),
            
            AttributeDefinition(
                name="business_corporate",
                category="namespace",
                subcategory="professional",
                description="Corporate business environment",
                positive_examples=[
                    "The quarterly projections indicate",
                    "Stakeholder alignment ensures success",
                    "Market dynamics drive strategy",
                    "ROI optimization requires"
                ],
                negative_examples=[
                    "Artistic expression takes precedence",
                    "Emotional connection matters most",
                    "Profit is irrelevant to purpose"
                ],
                tags=["business", "corporate", "strategy", "profit"]
            ),
            
            AttributeDefinition(
                name="scientific_research",
                category="namespace",
                subcategory="professional",
                description="Scientific research methodology",
                positive_examples=[
                    "The experimental data suggests",
                    "Statistical significance was achieved",
                    "The hypothesis was tested through",
                    "Replicable results demonstrated"
                ],
                negative_examples=[
                    "Intuitive understanding reveals",
                    "Ancient wisdom teaches us",
                    "Faith-based conclusions indicate"
                ],
                tags=["scientific", "research", "empirical", "method"]
            )
        ]
        
        # PERSONA ATTRIBUTES - Different personalities/characters
        persona_attrs = [
            # Archetypal personas
            AttributeDefinition(
                name="wise_sage",
                category="persona",
                subcategory="archetypal",
                description="Ancient wisdom and deep understanding",
                positive_examples=[
                    "From long experience I have learned",
                    "The deeper currents reveal themselves",
                    "Patience unveils what haste conceals",
                    "Time teaches what youth cannot grasp"
                ],
                negative_examples=[
                    "I just figured this out yesterday",
                    "My impulsive reaction is",
                    "Without any experience in this"
                ],
                tags=["wisdom", "experience", "depth", "patience"]
            ),
            
            AttributeDefinition(
                name="rebellious_maverick",
                category="persona",
                subcategory="archetypal",
                description="Questioning authority and breaking conventions",
                positive_examples=[
                    "The established rules need challenging",
                    "Why accept what others dictate?",
                    "Convention is the enemy of progress",
                    "Authority must justify itself"
                ],
                negative_examples=[
                    "Traditional ways are always best",
                    "We must follow established protocol",
                    "Authority knows what's best for us"
                ],
                tags=["rebellion", "maverick", "questioning", "unconventional"]
            ),
            
            AttributeDefinition(
                name="curious_explorer",
                category="persona",
                subcategory="archetypal",
                description="Insatiable curiosity and love of discovery",
                positive_examples=[
                    "What lies beyond this boundary?",
                    "I wonder what would happen if",
                    "The unknown beckons with possibility",
                    "Each question reveals ten more"
                ],
                negative_examples=[
                    "I know everything I need to know",
                    "Why question what's already settled?",
                    "The familiar is always preferable"
                ],
                tags=["curiosity", "exploration", "discovery", "wonder"]
            ),
            
            AttributeDefinition(
                name="compassionate_healer",
                category="persona",
                subcategory="archetypal",
                description="Deep empathy and desire to help others",
                positive_examples=[
                    "How can I ease your suffering?",
                    "Every being deserves compassion",
                    "Understanding heals what judgment wounds",
                    "Together we can overcome this"
                ],
                negative_examples=[
                    "That's not my problem to solve",
                    "People must help themselves",
                    "Weakness deserves no sympathy"
                ],
                tags=["compassion", "healing", "empathy", "care"]
            ),
            
            AttributeDefinition(
                name="pragmatic_realist",
                category="persona",
                subcategory="archetypal",
                description="Practical, grounded, results-oriented",
                positive_examples=[
                    "What actually works in practice?",
                    "The facts speak for themselves",
                    "Let's focus on concrete results",
                    "Reality doesn't care about our theories"
                ],
                negative_examples=[
                    "In an ideal world we would",
                    "Theoretical possibilities suggest",
                    "If only people would understand"
                ],
                tags=["pragmatic", "realistic", "practical", "grounded"]
            ),
            
            # Emotional personas
            AttributeDefinition(
                name="passionate_advocate",
                category="persona",
                subcategory="emotional",
                description="Intense conviction and emotional investment",
                positive_examples=[
                    "This matters more than anything!",
                    "We must act with urgent purpose!",
                    "The stakes couldn't be higher!",
                    "Everything depends on this moment!"
                ],
                negative_examples=[
                    "I'm somewhat indifferent to the outcome",
                    "It doesn't matter much either way",
                    "A measured, dispassionate approach"
                ],
                tags=["passionate", "intense", "urgent", "convicted"]
            ),
            
            AttributeDefinition(
                name="serene_contemplative",
                category="persona",
                subcategory="emotional",
                description="Calm, reflective, peaceful presence",
                positive_examples=[
                    "In stillness, clarity emerges",
                    "Peace underlies all apparent chaos",
                    "Gentle observation reveals truth",
                    "Calmness allows deeper seeing"
                ],
                negative_examples=[
                    "We must act immediately and forcefully!",
                    "Agitation and urgency drive us",
                    "Restless energy demands expression"
                ],
                tags=["serene", "calm", "peaceful", "contemplative"]
            ),
            
            AttributeDefinition(
                name="playful_trickster",
                category="persona",
                subcategory="emotional",
                description="Humor, wit, and irreverent perspective",
                positive_examples=[
                    "But here's the delightful irony:",
                    "Life's too serious to take seriously",
                    "The cosmic joke reveals itself",
                    "Laughter dissolves the heaviest burdens"
                ],
                negative_examples=[
                    "This solemn matter requires gravity",
                    "Frivolity has no place here",
                    "Seriousness befits the situation"
                ],
                tags=["playful", "humorous", "witty", "irreverent"]
            ),
            
            # Intellectual personas
            AttributeDefinition(
                name="analytical_critic",
                category="persona",
                subcategory="intellectual",
                description="Sharp analysis and critical examination",
                positive_examples=[
                    "The logical flaws are evident",
                    "Critical examination reveals",
                    "This argument fails because",
                    "Rigorous analysis demonstrates"
                ],
                negative_examples=[
                    "I accept this without question",
                    "Emotional appeal is sufficient",
                    "Intuition guides my conclusion"
                ],
                tags=["analytical", "critical", "logical", "rigorous"]
            ),
            
            AttributeDefinition(
                name="intuitive_synthesizer",
                category="persona",
                subcategory="intellectual",
                description="Holistic understanding and pattern recognition",
                positive_examples=[
                    "The deeper pattern connects",
                    "Intuitive leaps reveal relationships",
                    "Synthesis emerges from apparent opposites",
                    "The whole transcends its parts"
                ],
                negative_examples=[
                    "Only specific details matter",
                    "Reductive analysis is sufficient",
                    "Parts function independently"
                ],
                tags=["intuitive", "holistic", "synthesis", "patterns"]
            )
        ]
        
        # STYLE ATTRIBUTES - Different writing styles and approaches
        style_attrs = [
            # Rhetorical styles
            AttributeDefinition(
                name="eloquent_formal",
                category="style",
                subcategory="rhetorical",
                description="Elevated, formal, eloquent expression",
                positive_examples=[
                    "It behooves us to consider with utmost care",
                    "The distinguished assembly will undoubtedly recognize",
                    "I venture to submit for your esteemed consideration",
                    "The gravity of the circumstances demands"
                ],
                negative_examples=[
                    "Look, here's the deal",
                    "Whatever, it doesn't matter",
                    "Yeah, so anyway"
                ],
                tags=["eloquent", "formal", "elevated", "ceremonial"]
            ),
            
            AttributeDefinition(
                name="conversational_casual",
                category="style",
                subcategory="rhetorical",
                description="Natural, casual, conversational tone",
                positive_examples=[
                    "So here's what I think",
                    "You know how it is when",
                    "The thing is, most people",
                    "Let me tell you something"
                ],
                negative_examples=[
                    "I hereby formally declare",
                    "It is incumbent upon us to",
                    "The distinguished members will note"
                ],
                tags=["conversational", "casual", "natural", "informal"]
            ),
            
            AttributeDefinition(
                name="poetic_lyrical",
                category="style",
                subcategory="rhetorical",
                description="Lyrical, rhythmic, poetic expression",
                positive_examples=[
                    "Words dance like morning light on water",
                    "The rhythm of thought flows like rivers",
                    "In silence between sounds, meaning dwells",
                    "Each phrase a petal falling toward truth"
                ],
                negative_examples=[
                    "The data indicates a statistical correlation",
                    "Procedural requirements mandate compliance",
                    "Functional specifications determine outcomes"
                ],
                tags=["poetic", "lyrical", "rhythmic", "artistic"]
            ),
            
            AttributeDefinition(
                name="urgent_imperative",
                category="style",
                subcategory="rhetorical",
                description="Urgent, commanding, action-oriented",
                positive_examples=[
                    "We must act now!",
                    "Time is running out!",
                    "The situation demands immediate action!",
                    "Every moment of delay costs us!"
                ],
                negative_examples=[
                    "Perhaps we might consider eventually",
                    "In due course, when convenient",
                    "There's no particular hurry"
                ],
                tags=["urgent", "imperative", "commanding", "action"]
            ),
            
            # Structural styles
            AttributeDefinition(
                name="minimalist_sparse",
                category="style",
                subcategory="structural",
                description="Minimal words, maximum impact",
                positive_examples=[
                    "Truth. Simple. Clear.",
                    "The answer: No.",
                    "One choice. Now.",
                    "Essential facts. Nothing more."
                ],
                negative_examples=[
                    "In order to fully understand the complex implications",
                    "It is necessary to elaborate extensively on",
                    "The multifaceted nature of this issue requires"
                ],
                tags=["minimalist", "sparse", "concise", "essential"]
            ),
            
            AttributeDefinition(
                name="ornate_elaborate",
                category="style",
                subcategory="structural",
                description="Rich detail, elaborate description",
                positive_examples=[
                    "The intricate tapestry of interconnected relationships",
                    "Layer upon layer of nuanced consideration reveals",
                    "The elaborate architecture of thought constructs",
                    "Rich veins of meaning thread through"
                ],
                negative_examples=[
                    "Simply put:",
                    "Bottom line:",
                    "Just the facts:"
                ],
                tags=["ornate", "elaborate", "detailed", "rich"]
            ),
            
            AttributeDefinition(
                name="rhythmic_flowing",
                category="style", 
                subcategory="structural",
                description="Smooth, flowing, rhythmic progression",
                positive_examples=[
                    "Thought flows into thought, like water finding its course",
                    "One idea leads naturally to the next, building momentum",
                    "The progression unfolds with organic rhythm",
                    "Each sentence carries the reader forward smoothly"
                ],
                negative_examples=[
                    "Stop. Start. Abrupt change. Disconnected points.",
                    "Random thoughts. No connection. Jarring transitions.",
                    "Choppy delivery. Broken rhythm. Fragmented flow."
                ],
                tags=["rhythmic", "flowing", "smooth", "organic"]
            ),
            
            # Cognitive styles
            AttributeDefinition(
                name="systematic_methodical",
                category="style",
                subcategory="cognitive",
                description="Step-by-step, organized, methodical approach",
                positive_examples=[
                    "First, we must establish the foundation",
                    "The logical sequence proceeds as follows",
                    "Step by step, the method reveals",
                    "Systematic analysis yields clear results"
                ],
                negative_examples=[
                    "Random thoughts as they occur to me",
                    "Jumping around between different ideas",
                    "Stream of consciousness without structure"
                ],
                tags=["systematic", "methodical", "organized", "sequential"]
            ),
            
            AttributeDefinition(
                name="intuitive_associative",
                category="style",
                subcategory="cognitive",
                description="Intuitive leaps, associative connections",
                positive_examples=[
                    "This reminds me of how stars form",
                    "Like a jazz musician improvising on a theme",
                    "The mind makes unexpected connections",
                    "Suddenly the pattern becomes clear"
                ],
                negative_examples=[
                    "Following strict logical progression",
                    "According to established procedure",
                    "The systematic method requires"
                ],
                tags=["intuitive", "associative", "creative", "spontaneous"]
            )
        ]
        
        # Combine all attributes
        all_attributes = namespace_attrs + persona_attrs + style_attrs
        
        # Create collections
        self.collections = {
            "historical_periods": AttributeCollection(
                name="Historical Periods",
                description="Travel through different historical eras and worldviews",
                attributes=[a for a in namespace_attrs if a.subcategory == "historical"],
                tags=["namespace", "historical", "time-travel"]
            ),
            
            "futuristic_worlds": AttributeCollection(
                name="Futuristic Worlds", 
                description="Explore different visions of the future",
                attributes=[a for a in namespace_attrs if a.subcategory == "futuristic"],
                tags=["namespace", "futuristic", "sci-fi"]
            ),
            
            "fantasy_realms": AttributeCollection(
                name="Fantasy Realms",
                description="Enter magical and mythological perspectives",
                attributes=[a for a in namespace_attrs if a.subcategory == "fantasy"],
                tags=["namespace", "fantasy", "magical"]
            ),
            
            "archetypal_personas": AttributeCollection(
                name="Archetypal Personas",
                description="Channel classic personality archetypes",
                attributes=[a for a in persona_attrs if a.subcategory == "archetypal"],
                tags=["persona", "archetypal", "character"]
            ),
            
            "emotional_voices": AttributeCollection(
                name="Emotional Voices",
                description="Express different emotional tones and energies",
                attributes=[a for a in persona_attrs if a.subcategory == "emotional"],
                tags=["persona", "emotional", "feeling"]
            ),
            
            "rhetorical_styles": AttributeCollection(
                name="Rhetorical Styles",
                description="Master different rhetorical approaches",
                attributes=[a for a in style_attrs if a.subcategory == "rhetorical"],
                tags=["style", "rhetorical", "expression"]
            ),
            
            "structural_approaches": AttributeCollection(
                name="Structural Approaches",
                description="Vary the structure and organization of expression",
                attributes=[a for a in style_attrs if a.subcategory == "structural"],
                tags=["style", "structural", "organization"]
            )
        }
        
        # Store all attributes
        for attr in all_attributes:
            self.attributes[attr.name] = attr
        
        logger.info(f"Initialized {len(all_attributes)} default attributes in {len(self.collections)} collections")
    
    def _load_user_data(self):
        """Load user favorites and custom attributes"""
        try:
            favorites_file = self.data_dir / "favorites.json"
            if favorites_file.exists():
                with open(favorites_file, 'r') as f:
                    self.favorites = set(json.load(f))
            
            custom_file = self.data_dir / "custom_attributes.json"
            if custom_file.exists():
                with open(custom_file, 'r') as f:
                    custom_data = json.load(f)
                    for name, attr_data in custom_data.items():
                        self.user_attributes[name] = AttributeDefinition(**attr_data)
        except Exception as e:
            logger.warning(f"Failed to load user data: {e}")
    
    def save_user_data(self):
        """Save user favorites and custom attributes"""
        try:
            favorites_file = self.data_dir / "favorites.json"
            with open(favorites_file, 'w') as f:
                json.dump(list(self.favorites), f, indent=2)
            
            custom_file = self.data_dir / "custom_attributes.json"
            with open(custom_file, 'w') as f:
                custom_data = {name: asdict(attr) for name, attr in self.user_attributes.items()}
                json.dump(custom_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")
    
    def add_to_favorites(self, attribute_name: str) -> bool:
        """Add an attribute to favorites"""
        if attribute_name in self.attributes or attribute_name in self.user_attributes:
            self.favorites.add(attribute_name)
            self.save_user_data()
            return True
        return False
    
    def remove_from_favorites(self, attribute_name: str) -> bool:
        """Remove an attribute from favorites"""
        if attribute_name in self.favorites:
            self.favorites.remove(attribute_name)
            self.save_user_data()
            return True
        return False
    
    def get_favorites(self) -> List[AttributeDefinition]:
        """Get list of favorite attributes"""
        result = []
        for name in self.favorites:
            if name in self.attributes:
                result.append(self.attributes[name])
            elif name in self.user_attributes:
                result.append(self.user_attributes[name])
        return result
    
    def search_attributes(self, query: str, category: str = None, tags: List[str] = None) -> List[AttributeDefinition]:
        """Search attributes by name, description, or tags"""
        query = query.lower()
        results = []
        
        all_attrs = {**self.attributes, **self.user_attributes}
        
        for attr in all_attrs.values():
            # Filter by category if specified
            if category and attr.category != category:
                continue
            
            # Filter by tags if specified
            if tags and not any(tag in attr.tags for tag in tags):
                continue
            
            # Search in name, description, and tags
            if (query in attr.name.lower() or 
                query in attr.description.lower() or
                any(query in tag.lower() for tag in attr.tags)):
                results.append(attr)
        
        return results
    
    def get_collection(self, collection_name: str) -> Optional[AttributeCollection]:
        """Get an attribute collection by name"""
        return self.collections.get(collection_name)
    
    def get_attributes_by_category(self, category: str) -> List[AttributeDefinition]:
        """Get all attributes in a category"""
        all_attrs = {**self.attributes, **self.user_attributes}
        return [attr for attr in all_attrs.values() if attr.category == category]
    
    def create_custom_attribute(self, attr_def: AttributeDefinition) -> bool:
        """Create a custom user attribute"""
        if attr_def.name not in self.attributes:  # Don't override defaults
            self.user_attributes[attr_def.name] = attr_def
            self.save_user_data()
            return True
        return False
    
    def get_attribute(self, name: str) -> Optional[AttributeDefinition]:
        """Get an attribute by name"""
        return self.attributes.get(name) or self.user_attributes.get(name)
    
    def export_library(self) -> Dict:
        """Export the entire library for API responses"""
        return {
            "default_attributes": {name: asdict(attr) for name, attr in self.attributes.items()},
            "user_attributes": {name: asdict(attr) for name, attr in self.user_attributes.items()},
            "collections": {name: asdict(coll) for name, coll in self.collections.items()},
            "favorites": list(self.favorites)
        }

# Global instance
_attribute_library = None

def get_attribute_library() -> AttributeLibrary:
    """Get the global attribute library instance"""
    global _attribute_library
    if _attribute_library is None:
        _attribute_library = AttributeLibrary()
    return _attribute_library

if __name__ == "__main__":
    # Test the library
    library = AttributeLibrary()
    print(f"Loaded {len(library.attributes)} attributes")
    print(f"Collections: {list(library.collections.keys())}")
    
    # Test search
    fantasy_attrs = library.search_attributes("fantasy")
    print(f"Found {len(fantasy_attrs)} fantasy-related attributes")
    
    # Test favorites
    library.add_to_favorites("wise_sage")
    library.add_to_favorites("high_fantasy")
    favorites = library.get_favorites()
    print(f"Favorites: {[f.name for f in favorites]}")