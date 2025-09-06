# mages_gambit_solver.py

import logging

logger = logging.getLogger(__name__)

def solve_mages_gambit(intel, reserve, fronts, stamina):
    """
    Solve the mage's gambit problem for Klein Moretti.
    
    Args:
        intel: List of [front, mana_cost] pairs representing undead attacks in sequence
        reserve: Maximum mana Klein has
        fronts: Number of fronts (not used in calculation but part of input)
        stamina: Maximum number of spells Klein can cast before needing cooldown
    
    Returns:
        int: Minimum time in minutes for Klein to defeat all undead and be ready for expedition
    """
    logger.info(f"Solving mages gambit: intel={intel}, reserve={reserve}, fronts={fronts}, stamina={stamina}")
    
    current_mp = reserve
    current_stamina = stamina
    total_time = 0
    i = 0
    
    while i < len(intel):
        current_front = intel[i][0]
        consecutive_attacks = []
        
        # Collect consecutive attacks on the same front
        while i < len(intel) and intel[i][0] == current_front:
            consecutive_attacks.append(intel[i][1])  # mana cost
            i += 1
        
        logger.info(f"Processing front {current_front} with attacks requiring: {consecutive_attacks} MP")
        
        # Process this batch of consecutive attacks
        batch_result = process_batch(consecutive_attacks, current_mp, current_stamina, reserve, stamina)
        total_time += batch_result['time']
        current_mp = batch_result['final_mp']
        current_stamina = batch_result['final_stamina']
        
        logger.info(f"Batch completed in {batch_result['time']} minutes. Total time: {total_time}")
    
    # Final cooldown to be ready for expedition
    total_time += 10
    logger.info(f"Final cooldown: +10 minutes. Total time: {total_time}")
    
    return total_time

def process_batch(mana_costs, current_mp, current_stamina, max_mp, max_stamina):
    """
    Process a batch of consecutive attacks on the same front.
    
    Returns:
        dict: {'time': int, 'final_mp': int, 'final_stamina': int}
    """
    total_time = 0
    mp = current_mp
    stamina = current_stamina
    attack_index = 0
    is_first_attack_in_location = True
    
    while attack_index < len(mana_costs):
        mana_cost = mana_costs[attack_index]
        
        # Check if cooldown is needed
        if mp < mana_cost or stamina == 0:
            total_time += 10  # cooldown
            mp = max_mp
            stamina = max_stamina
            is_first_attack_in_location = True  # After cooldown, need to set new target
            logger.debug(f"Cooldown needed: +10 min (MP={mp}, Stamina={stamina})")
        
        # Perform the attack
        mp -= mana_cost
        stamina -= 1
        
        # Time calculation
        if is_first_attack_in_location:
            total_time += 10  # First attack at this location takes 10 minutes
            is_first_attack_in_location = False
            logger.debug(f"New target attack ({mana_cost} MP): +10 min (MP={mp}, Stamina={stamina})")
        else:
            # Extending AOE takes 0 time
            logger.debug(f"Extend AOE ({mana_cost} MP): +0 min (MP={mp}, Stamina={stamina})")
        
        attack_index += 1
    
    return {
        'time': total_time,
        'final_mp': mp,
        'final_stamina': stamina
    }

def solve_mages_gambit_multiple(test_cases):
    """
    Solve multiple mages gambit test cases.
    
    Args:
        test_cases: List of dictionaries with keys: intel, reserve, fronts, stamina
    
    Returns:
        List of dictionaries with key: time
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Processing test case {i+1}/{len(test_cases)}")
        
        try:
            intel = test_case['intel']
            reserve = test_case['reserve']
            fronts = test_case['fronts']
            stamina = test_case['stamina']
            
            time = solve_mages_gambit(intel, reserve, fronts, stamina)
            results.append({'time': time})
            
        except KeyError as e:
            logger.error(f"Missing required field in test case {i+1}: {e}")
            raise ValueError(f"Test case {i+1} missing required field: {e}")
        except Exception as e:
            logger.error(f"Error processing test case {i+1}: {e}")
            raise
    
    return results