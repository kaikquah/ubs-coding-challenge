import re
import logging

logger = logging.getLogger(__name__)

# Precompiled regex patterns for better performance
ROMAN_PATTERN = re.compile(r'^[IVXLCDM]+$')
ARABIC_PATTERN = re.compile(r'^\d+$')

def parse_roman(s):
    """
    Parse Roman numerals to integer.
    Time: O(n) where n is length of string
    Space: O(1)
    """
    if not s:
        return 0
        
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    prev_value = 0
    
    # Process from right to left for subtractive notation
    for char in reversed(s):
        value = roman_values.get(char, 0)
        if value == 0:  # Invalid Roman character
            return 0
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total

def is_roman(s):
    """Check if string is a valid Roman numeral."""
    return bool(ROMAN_PATTERN.match(s))

def parse_english(s):
    """
    Parse English number words to integer.
    Handles cases like "one hundred and twenty"
    Time: O(n) where n is number of words
    Space: O(n) for word splitting
    """
    if not s:
        return 0
        
    # Normalize: remove "and", handle case
    s = s.lower().replace(" and ", " ").strip()
    
    # Handle zero special case
    if s == "zero":
        return 0
    
    # Define word to number mappings
    ones = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
    }
    
    tens = {
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    
    scales = {
        'hundred': 100,
        'thousand': 1000,
        'million': 1000000,
        'billion': 1000000000
    }
    
    words = s.split()
    if not words:
        return 0
        
    total = 0
    current = 0
    
    for word in words:
        if word in ones:
            current += ones[word]
        elif word in tens:
            current += tens[word]
        elif word in scales:
            scale = scales[word]
            if scale == 100:
                if current == 0:
                    current = 1  # "hundred" means "one hundred"
                current *= scale
            else:  # thousand, million, billion
                if current == 0:
                    current = 1  # "thousand" means "one thousand"
                total += current * scale
                current = 0
        # Ignore unknown words (could be "and" remnants)
    
    return total + current

def is_english(s):
    """Check if string is English number words."""
    if not s:
        return False
        
    english_words = {
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
        'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 
        'billion', 'and'
    }
    
    words = s.lower().replace(',', '').split()
    return len(words) > 0 and all(word in english_words for word in words)

def parse_chinese(s):
    """
    Parse Chinese numerals (both Traditional and Simplified) to integer.
    Handles mixed traditional/simplified, large numbers up to 億/亿
    Time: O(n) where n is length of string
    Space: O(1)
    """
    if not s:
        return 0
    
    # Check if it's pure Arabic numerals
    if ARABIC_PATTERN.match(s):
        return int(s)
    
    # Comprehensive digit mappings
    digit_map = {
        # Zeros
        '零': 0, '〇': 0, '０': 0, '0': 0,
        # Ones
        '一': 1, '壹': 1, '１': 1, '1': 1,
        # Twos (including 两/兩 used in some contexts)
        '二': 2, '貳': 2, '贰': 2, '２': 2, '2': 2, '两': 2, '兩': 2,
        # Threes
        '三': 3, '參': 3, '叁': 3, '３': 3, '3': 3,
        # Fours
        '四': 4, '肆': 4, '４': 4, '4': 4,
        # Fives
        '五': 5, '伍': 5, '５': 5, '5': 5,
        # Sixes
        '六': 6, '陸': 6, '陆': 6, '６': 6, '6': 6,
        # Sevens
        '七': 7, '柒': 7, '７': 7, '7': 7,
        # Eights
        '八': 8, '捌': 8, '８': 8, '8': 8,
        # Nines
        '九': 9, '玖': 9, '９': 9, '9': 9,
        # Place values
        '十': 10, '拾': 10,
        '百': 100, '佰': 100,
        '千': 1000, '仟': 1000,
        '万': 10000, '萬': 10000,
        '億': 100000000, '亿': 100000000
    }
    
    result = 0
    temp = 0
    section = 0
    last_was_digit = False
    
    for i, char in enumerate(s):
        if char not in digit_map:
            continue
            
        num = digit_map[char]
        
        if num >= 100000000:  # 億/亿
            section = (section + temp) * num
            result += section
            section = 0
            temp = 0
            last_was_digit = False
        elif num >= 10000:  # 万/萬
            section = (section + temp) * num
            temp = 0
            last_was_digit = False
        elif num >= 10:  # 十, 百, 千
            if temp == 0:
                temp = 1  # Handle cases like "十" = 10, "百" = 100
            temp *= num
            last_was_digit = False
        else:  # 0-9
            if last_was_digit:
                temp = temp * 10 + num  # Handle consecutive digits
            else:
                temp += num
            last_was_digit = True
    
    return result + section + temp

def is_chinese(s):
    """Check if string contains Chinese numerals."""
    if not s:
        return False
        
    # Comprehensive set of Chinese number characters
    chinese_chars = set('零〇０一壹１二貳贰２两兩三參叁３四肆４五伍５六陸陆６七柒７八捌８九玖９十拾百佰千仟万萬億亿')
    return any(char in chinese_chars for char in s)

def parse_german(s):
    """
    Parse German number words to integer.
    Handles compound words with 'und', 'hundert', 'tausend'
    Time: O(n) where n is length of string
    Space: O(1)
    """
    if not s:
        return 0
        
    s = s.lower().strip()
    
    # German number mappings
    ones_de = {
        'null': 0, 'eins': 1, 'ein': 1, 'eine': 1, 'zwei': 2, 'drei': 3, 'vier': 4,
        'fünf': 5, 'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9,
        'zehn': 10, 'elf': 11, 'zwölf': 12, 'dreizehn': 13,
        'vierzehn': 14, 'fünfzehn': 15, 'sechzehn': 16,
        'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19
    }
    
    tens_de = {
        'zwanzig': 20, 'dreißig': 30, 'dreissig': 30, 'vierzig': 40, 
        'fünfzig': 50, 'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90
    }
    
    # Handle 'million'
    if 'million' in s:
        parts = s.split('million')
        millions = parts[0].strip() if parts[0] else 'eine'
        millions_val = ones_de.get(millions, 1)
        remainder_val = parse_german(parts[1]) if len(parts) > 1 and parts[1] else 0
        return millions_val * 1000000 + remainder_val
    
    # Handle 'tausend' (thousand)
    if 'tausend' in s:
        parts = s.split('tausend', 1)
        thousands = parts[0].strip() if parts[0] else 'ein'
        
        # Parse the thousands part
        if thousands in ones_de:
            thousands_val = ones_de[thousands]
        elif 'hundert' in thousands:
            thousands_val = parse_german(thousands)
        else:
            thousands_val = 1
            
        remainder_val = parse_german(parts[1]) if len(parts) > 1 and parts[1] else 0
        return thousands_val * 1000 + remainder_val
    
    # Handle 'hundert' (hundred)
    if 'hundert' in s:
        parts = s.split('hundert', 1)
        hundreds = parts[0].strip() if parts[0] else 'ein'
        hundreds_val = ones_de.get(hundreds, 1)
        
        remainder = parts[1].strip() if len(parts) > 1 else ''
        if remainder:
            # Parse remainder which might have 'und'
            remainder_val = parse_german(remainder)
            return hundreds_val * 100 + remainder_val
        return hundreds_val * 100
    
    # Handle 'und' compounds (e.g., "siebenundachtzig" = 87)
    if 'und' in s:
        parts = s.split('und')
        if len(parts) == 2:
            ones_part = parts[0].strip()
            tens_part = parts[1].strip()
            
            ones_val = ones_de.get(ones_part, 0)
            tens_val = tens_de.get(tens_part, 0)
            
            if ones_val and tens_val:
                return ones_val + tens_val
    
    # Direct lookup
    if s in ones_de:
        return ones_de[s]
    if s in tens_de:
        return tens_de[s]
    
    return 0

def is_german(s):
    """Check if string is German number words."""
    if not s:
        return False
        
    s = s.lower()
    
    # Quick checks for German-specific patterns
    if any(keyword in s for keyword in ['und', 'hundert', 'tausend', 'million']):
        return True
    
    german_words = {
        'null', 'ein', 'eine', 'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 
        'sieben', 'acht', 'neun', 'zehn', 'elf', 'zwölf', 'dreizehn', 'vierzehn', 
        'fünfzehn', 'sechzehn', 'siebzehn', 'achtzehn', 'neunzehn', 'zwanzig', 
        'dreißig', 'dreissig', 'vierzig', 'fünfzig', 'sechzig', 'siebzig', 
        'achtzig', 'neunzig'
    }
    
    # Check if any word is German
    words = s.replace('und', ' ').replace('hundert', ' ').replace('tausend', ' ').split()
    return any(word in german_words for word in words)

def parse_number(s):
    """
    Parse a number string in any supported language to integer.
    Returns 0 if unable to parse.
    Time: O(n) where n is length of string
    """
    if not s or not s.strip():
        return 0
    
    s = s.strip()
    
    # Check Arabic numerals first (most common and fastest)
    if ARABIC_PATTERN.match(s):
        # Handle potential leading zeros
        return int(s)
    
    # Check Roman numerals
    if is_roman(s):
        return parse_roman(s)
    
    # Check German (before English due to potential overlap)
    if is_german(s):
        return parse_german(s)
    
    # Check English
    if is_english(s):
        return parse_english(s)
    
    # Check Chinese (Traditional or Simplified)
    if is_chinese(s):
        return parse_chinese(s)
    
    # Log warning for unparseable input
    logger.warning(f"Could not parse: '{s}'")
    return 0

def get_language_priority(s):
    """
    Get the language priority for sorting duplicates.
    Order: Roman, English, Traditional Chinese, Simplified Chinese, German, Arabic
    Time: O(1) to O(n) depending on checks needed
    """
    if not s:
        return 7  # Unknown, lowest priority
        
    if ARABIC_PATTERN.match(s):
        return 6
    elif is_roman(s):
        return 1
    elif is_english(s):
        return 2
    elif is_german(s):
        return 5
    elif is_chinese(s):
        # Distinguish between Traditional and Simplified
        # Traditional-only characters
        traditional_only = set('萬億貳參陸')
        # Simplified-only characters  
        simplified_only = set('万亿贰叁陆')
        
        has_traditional = any(c in traditional_only for c in s)
        has_simplified = any(c in simplified_only for c in s)
        
        if has_traditional and not has_simplified:
            return 3  # Traditional Chinese
        elif has_simplified and not has_traditional:
            return 4  # Simplified Chinese
        else:
            # Ambiguous or uses shared characters
            # Default to simplified priority
            return 4
    
    return 7  # Unknown

def solve_duolingo_sort(data):
    """
    Solve the Duolingo Sort challenge.
    Time Complexity: O(n * m + n log n) where n is list size, m is avg string length
    Space Complexity: O(n) for the sorted list
    """
    try:
        part = data.get('part', '')
        challenge_input = data.get('challengeInput', {})
        unsorted_list = challenge_input.get('unsortedList', [])
        
        # Handle empty list edge case
        if not unsorted_list:
            return {'sortedList': []}
        
        # Handle single element
        if len(unsorted_list) == 1:
            if part == 'ONE':
                value = parse_number(unsorted_list[0])
                return {'sortedList': [str(value)]}
            else:
                return {'sortedList': unsorted_list}
        
        if part == 'ONE':
            # Part 1: Convert to integers and return as strings
            parsed_values = []
            for item in unsorted_list:
                value = parse_number(item)
                parsed_values.append(value)
            
            # Sort numerically
            parsed_values.sort()
            
            # Convert back to strings
            sorted_list = [str(v) for v in parsed_values]
            
        else:  # part == 'TWO'
            # Part 2: Keep original representations, sort by value then language
            items_with_metadata = []
            
            for item in unsorted_list:
                value = parse_number(item)
                priority = get_language_priority(item)
                items_with_metadata.append((value, priority, item))
            
            # Sort by value first, then by language priority
            items_with_metadata.sort(key=lambda x: (x[0], x[1]))
            
            # Extract sorted items
            sorted_list = [item[2] for item in items_with_metadata]
        
        return {'sortedList': sorted_list}
        
    except Exception as e:
        logger.error(f"Error in solve_duolingo_sort: {e}")
        # Return empty list on error to avoid breaking
        return {'sortedList': []}