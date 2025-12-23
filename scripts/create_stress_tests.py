"""
Create stress test datasets for robustness evaluation
"""
import json
from pathlib import Path
from loguru import logger

def create_negation_tests():
    """Test cases with negation (models often fail here)"""
    
    tests = [
        # Double negatives (should be positive)
        {"text": "This movie was not bad at all, actually quite enjoyable.", "label": 1, "type": "double_negative"},
        {"text": "I can't say I didn't like it. It was pretty good.", "label": 1, "type": "double_negative"},
        {"text": "It's not that the movie wasn't entertaining.", "label": 1, "type": "double_negative"},
        
        # Simple negations (should be opposite of key word)
        {"text": "The movie was not good. Very disappointing.", "label": 0, "type": "simple_negation"},
        {"text": "Not terrible, but definitely not great either.", "label": 0, "type": "simple_negation"},
        {"text": "This film was not worth watching.", "label": 0, "type": "simple_negation"},
        
        # Never/no negations
        {"text": "Never have I seen such a boring film.", "label": 0, "type": "never_negation"},
        {"text": "No amount of good acting could save this plot.", "label": 0, "type": "no_negation"},
        {"text": "I have never been so entertained by a movie.", "label": 1, "type": "never_negation"},
        
        # Negation with positive words
        {"text": "Not the best movie I've ever seen, but still enjoyable.", "label": 1, "type": "qualified_positive"},
        {"text": "It's not perfect, but it's a solid film.", "label": 1, "type": "qualified_positive"},
        {"text": "Not excellent, but definitely not bad.", "label": 1, "type": "qualified_positive"},
        
        # Negation with negative words (should be positive)
        {"text": "Not disappointing at all. I really enjoyed it.", "label": 1, "type": "negated_negative"},
        {"text": "Definitely not a waste of time. Highly recommended.", "label": 1, "type": "negated_negative"},
        {"text": "I wasn't bored for a single moment.", "label": 1, "type": "negated_negative"},
        
        # Complex negations
        {"text": "I wouldn't say I didn't enjoy it, but it wasn't great.", "label": 0, "type": "complex_negation"},
        {"text": "Can't complain about the acting, but the plot was weak.", "label": 0, "type": "complex_negation"},
        
        # Additional challenging cases
        {"text": "Nothing special, but not terrible either. Just okay.", "label": 0, "type": "neutral_negative"},
        {"text": "I can't say anything bad about this film. Loved it!", "label": 1, "type": "cant_say_bad"},
        {"text": "Not once did I think about leaving the theater.", "label": 1, "type": "not_once"},
        {"text": "There's nothing I didn't like about this movie.", "label": 1, "type": "nothing_didnt_like"},
        
        # Edge cases
        {"text": "Not good, not bad, just meh.", "label": 0, "type": "truly_neutral"},
        {"text": "I'm not unhappy I watched it.", "label": 1, "type": "not_unhappy"},
        {"text": "It's not that I hated it, I just didn't like it.", "label": 0, "type": "not_that"},
        {"text": "I wouldn't not recommend it.", "label": 1, "type": "wouldnt_not"},
        {"text": "Not the worst movie ever, but close.", "label": 0, "type": "not_worst_but"},
    ]
    
    return tests

def create_sarcasm_tests():
    """Test cases with sarcasm (very challenging)"""
    
    tests = [
        # Obvious sarcasm
        {"text": "Oh great, another boring superhero movie. Just what we needed.", "label": 0, "type": "oh_great"},
        {"text": "Yeah, because the world really needed a fifth sequel.", "label": 0, "type": "yeah_because"},
        {"text": "Sure, this is definitely the best movie of the century. Not.", "label": 0, "type": "sure_definitely"},
        {"text": "Wow, what an original plot. Never seen that before!", "label": 0, "type": "wow_original"},
        
        # Ironic praise
        {"text": "Brilliant! I loved every one of the 3 hours I'll never get back.", "label": 0, "type": "time_back"},
        {"text": "Amazing! It made me want to walk out after 10 minutes.", "label": 0, "type": "walk_out"},
        {"text": "Fantastic acting. I could barely stay awake.", "label": 0, "type": "barely_awake"},
        
        # Hyperbolic sarcasm
        {"text": "Best movie ever! If you enjoy watching paint dry.", "label": 0, "type": "paint_dry"},
        {"text": "Absolutely riveting. I only checked my phone 47 times.", "label": 0, "type": "checked_phone"},
        {"text": "Oscar-worthy performance! Said no one ever.", "label": 0, "type": "said_no_one"},
        
        # Subtle sarcasm
        {"text": "Well, that was two hours well spent. I guess.", "label": 0, "type": "well_spent_guess"},
        {"text": "I'm so glad I paid money to see this masterpiece.", "label": 0, "type": "paid_money"},
        {"text": "Really lived up to the hype. Totally worth it.", "label": 0, "type": "lived_up_hype"},
        
        # Sarcasm with exclamation marks
        {"text": "Oh wow!!! Such an incredible story!!! Not predictable at all!!!", "label": 0, "type": "excessive_exclamation"},
        {"text": "Amazing!!! Every cliche in the book!!! So creative!!!", "label": 0, "type": "so_creative"},
        
        # Mixed with genuine criticism
        {"text": "Sure, the CGI was great. But everything else? Terrible.", "label": 0, "type": "sure_but"},
        {"text": "Yeah, I 'loved' it. Just like I 'love' root canals.", "label": 0, "type": "loved_quotes"},
    ]
    
    return tests

def create_ood_tests():
    """Out-of-distribution tests (different domains)"""
    
    tests = [
        # Medical/health reviews (should still work)
        {"text": "This medical documentary was incredibly informative and well-made.", "label": 1, "type": "medical"},
        {"text": "The documentary about vaccines was poorly researched and biased.", "label": 0, "type": "medical"},
        
        # Political content (different language)
        {"text": "This political documentary presents a compelling and balanced view.", "label": 1, "type": "political"},
        {"text": "Politically biased propaganda disguised as a documentary.", "label": 0, "type": "political"},
        
        # Educational content
        {"text": "An excellent educational film that teaches complex concepts clearly.", "label": 1, "type": "educational"},
        {"text": "Boring educational content that fails to engage the audience.", "label": 0, "type": "educational"},
        
        # Animation (different style)
        {"text": "The animation quality was stunning and the story was heartwarming.", "label": 1, "type": "animation"},
        {"text": "Poor animation and a childish plot that even kids won't enjoy.", "label": 0, "type": "animation"},
        
        # Foreign films (may have different review patterns)
        {"text": "This foreign film offers a unique cultural perspective and beautiful cinematography.", "label": 1, "type": "foreign"},
        {"text": "Slow-paced foreign film with subtitles that dragged on forever.", "label": 0, "type": "foreign"},
        
        # Short films (different expectations)
        {"text": "A brilliant short film that tells a complete story in 15 minutes.", "label": 1, "type": "short"},
        {"text": "This short film felt incomplete and poorly edited.", "label": 0, "type": "short"},
        
        # Classic films (different era language)
        {"text": "A timeless classic that holds up remarkably well today.", "label": 1, "type": "classic"},
        {"text": "This old film feels dated and the pacing is painfully slow.", "label": 0, "type": "classic"},
        
        # Experimental films
        {"text": "An innovative experimental film that pushes artistic boundaries.", "label": 1, "type": "experimental"},
        {"text": "Pretentious experimental nonsense that tries too hard to be artsy.", "label": 0, "type": "experimental"},
        
        # Technical reviews (focus on production quality)
        {"text": "The sound design and cinematography are technically flawless.", "label": 1, "type": "technical"},
        {"text": "Poor audio quality and shaky camera work ruined the experience.", "label": 0, "type": "technical"},
        
        # Streaming content (modern context)
        {"text": "Perfect binge-worthy content for a lazy weekend at home.", "label": 1, "type": "streaming"},
        {"text": "Not even worth the streaming service subscription fee.", "label": 0, "type": "streaming"},
    ]
    
    return tests

def save_stress_tests():
    """Generate and save all stress tests"""
    
    output_dir = Path("data/stress_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tests
    negation_tests = create_negation_tests()
    sarcasm_tests = create_sarcasm_tests()
    ood_tests = create_ood_tests()
    
    # Save each category
    datasets = {
        "negation": negation_tests,
        "sarcasm": sarcasm_tests,
        "ood": ood_tests
    }
    
    for name, tests in datasets.items():
        output_path = output_dir / f"{name}_test.json"
        with open(output_path, 'w') as f:
            json.dump(tests, f, indent=2)
        logger.info(f"💾 Saved {len(tests)} {name} tests to {output_path}")
    
    # Save combined
    combined = negation_tests + sarcasm_tests + ood_tests
    combined_path = output_dir / "all_stress_tests.json"
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    logger.info(f"💾 Saved {len(combined)} total tests to {combined_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("STRESS TEST DATASETS CREATED")
    print("="*80)
    print(f"Negation tests:  {len(negation_tests)}")
    print(f"Sarcasm tests:   {len(sarcasm_tests)}")
    print(f"OOD tests:       {len(ood_tests)}")
    print(f"Total:           {len(combined)}")
    print("="*80)
    
    return datasets

if __name__ == "__main__":
    save_stress_tests()