#!/usr/bin/env python3
"""
Demo of Multiple Rho-Conditioned Voice Styles
Shows different ways the same rho matrix can be used to generate responses
"""

import requests
import json
from rho_language_generator import EvolvingRhoLLM

def create_diverse_rho_states(generator):
    """Create different rho states with various text samples"""
    
    # Alice - childhood wonder and agency
    alice_text = ("Alice began to feel very sleepy, and she had just begun to dream about "
                 "the wonderful adventures that awaited her. She was no longer the timid girl "
                 "who had tumbled down the rabbit hole. Now she questioned everything and "
                 "stood up to the Queen of Hearts with surprising courage.")
    
    alice_response = requests.post(
        f"{generator.api_base_url}/rho/init",
        json={"seed_text": alice_text, "label": "alice_voice"}
    )
    alice_id = alice_response.json()["rho_id"] if alice_response.status_code == 200 else None
    
    # Kafka-esque existential text  
    kafka_text = ("The bureaucrat sat in his gray office, filling out forms that no one would read, "
                 "for purposes no one could explain. The meaninglessness pressed down like fog, "
                 "yet he continued, trapped in routines that defined his existence.")
    
    kafka_response = requests.post(
        f"{generator.api_base_url}/rho/init", 
        json={"seed_text": kafka_text, "label": "kafka_voice"}
    )
    kafka_id = kafka_response.json()["rho_id"] if kafka_response.status_code == 200 else None
    
    # Dickens-style social commentary
    dickens_text = ("The factory workers, men and women alike, toiled in conditions that would shame "
                   "the devils themselves. Yet among them was kindness, charity, and hope - the very "
                   "qualities that the wealthy masters above had lost in their pursuit of profit.")
    
    dickens_response = requests.post(
        f"{generator.api_base_url}/rho/init",
        json={"seed_text": dickens_text, "label": "dickens_voice"}  
    )
    dickens_id = dickens_response.json()["rho_id"] if dickens_response.status_code == 200 else None
    
    return {
        "alice": alice_id,
        "kafka": kafka_id, 
        "dickens": dickens_id
    }

def demonstrate_voice_styles():
    """Show how different rho states produce different voices"""
    generator = EvolvingRhoLLM()
    
    # Create diverse rho states
    voices = create_diverse_rho_states(generator)
    
    # Common query to test different perspectives
    query = "What does it mean to grow up and face the real world?"
    
    results = {}
    
    for voice_name, rho_id in voices.items():
        if rho_id:
            print(f"\n=== {voice_name.upper()} VOICE ===")
            
            # Test different response styles for each voice
            for style in ["analytical", "experiential", "synthetic"]:
                response = generator.generate_rho_conditioned_response(
                    query=query,
                    rho_id=rho_id, 
                    response_style=style
                )
                print(f"\n{style.title()} Style:")
                print(response)
                
                if voice_name not in results:
                    results[voice_name] = {}
                results[voice_name][style] = response
    
    return results

def demonstrate_quantum_attention():
    """Show quantum attention weights for different queries"""
    generator = EvolvingRhoLLM()
    
    # Create a test rho state
    create_response = requests.post(
        f"{generator.api_base_url}/rho/init",
        json={
            "seed_text": "The hero's journey took her through trials of courage, wisdom, and sacrifice. Each challenge revealed new depths of character.",
            "label": "hero_journey"
        }
    )
    
    if create_response.status_code != 200:
        return {"error": "Could not create test rho state"}
    
    rho_id = create_response.json()["rho_id"]
    rho_state = generator.load_rho_state(rho_id)
    
    if not rho_state:
        return {"error": "Could not load rho state"}
    
    queries = [
        "Tell me about courage and bravery",
        "What is the nature of wisdom?", 
        "How do we grow through sacrifice?",
        "What does it mean to be a hero?"
    ]
    
    attention_analysis = {}
    
    for query in queries:
        attention_weights = generator.quantum_attention.compute_attention_weights(
            query, rho_state
        )
        
        # Get top 3 most relevant attributes
        top_attributes = sorted(attention_weights.items(), 
                              key=lambda x: abs(x[1]), reverse=True)[:3]
        
        attention_analysis[query] = {
            "dominant_attributes": top_attributes,
            "interpretation": generator.interpret_dominant_attributes(top_attributes)
        }
        
        print(f"\n🔍 Query: '{query}'")
        print(f"   Attention: {top_attributes}")
        print(f"   Meaning: {attention_analysis[query]['interpretation']}")
    
    return attention_analysis

if __name__ == "__main__":
    print("🧠 RHO-CONDITIONED LANGUAGE GENERATION DEMO")
    print("=" * 50)
    
    print("\n1. TESTING DIFFERENT VOICE STYLES")
    voice_results = demonstrate_voice_styles()
    
    print("\n\n2. QUANTUM ATTENTION ANALYSIS") 
    attention_results = demonstrate_quantum_attention()
    
    print("\n\n✨ Demo complete! The rho matrices are learning to speak with distinct voices.")