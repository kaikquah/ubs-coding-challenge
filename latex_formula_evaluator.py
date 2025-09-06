import re
import math
import logging
from flask import Blueprint, jsonify, request
from typing import Dict, List, Union, Any, Optional

logger = logging.getLogger(__name__)

# Create Flask blueprint
latex_bp = Blueprint('latex_formula', __name__)


class LaTeXToken:
    """Represents a token in the LaTeX formula"""
    def __init__(self, type_: str, value: str, position: int = 0):
        self.type = type_
        self.value = value
        self.position = position
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"


class LaTeXTokenizer:
    """Tokenizes LaTeX mathematical expressions"""
    
    TOKEN_PATTERNS = [
        ('WHITESPACE', r'\s+'),
        ('DOLLAR', r'\$'),
        ('FRAC', r'\\frac'),
        ('MAX', r'\\max'),
        ('MIN', r'\\min'),
        ('SUM', r'\\sum'),
        ('LOG', r'\\log'),
        ('TEXT', r'\\text'),
        ('CDOT', r'\\cdot'),
        # Greek letters
        ('ALPHA', r'\\alpha'),
        ('BETA', r'\\beta'),
        ('GAMMA', r'\\gamma'),
        ('DELTA', r'\\delta'),
        ('SIGMA', r'\\sigma'),
        ('THETA', r'\\theta'),
        ('LAMBDA', r'\\lambda'),
        ('MU', r'\\mu'),
        ('PI', r'\\pi'),
        ('RHO', r'\\rho'),
        # Variable with brackets (like E[R_i])
        ('VAR_BRACKET', r'[A-Za-z]+\[[A-Za-z_0-9]+\]'),
        # Regular variable
        ('VARIABLE', r'[a-zA-Z][a-zA-Z0-9]*'),
        ('NUMBER', r'\d+\.?\d*'),
        ('PLUS', r'\+'),
        ('MINUS', r'-'),
        ('MULTIPLY', r'\*'),
        ('DIVIDE', r'/'),
        ('POWER', r'\^'),
        ('UNDERSCORE', r'_'),
        ('EQUALS', r'='),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACE', r'\{'),
        ('RBRACE', r'\}'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('COMMA', r','),
        ('E_CONST', r'(?<![a-zA-Z])e(?![a-zA-Z])'),  # Match 'e' only when not part of a word
        ('UNKNOWN', r'.'),
    ]
    
    def __init__(self):
        self.pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.TOKEN_PATTERNS)
        self.regex = re.compile(self.pattern)
    
    def tokenize(self, text: str) -> List[LaTeXToken]:
        """Convert LaTeX string to list of tokens"""
        tokens = []
        position = 0
        
        for match in self.regex.finditer(text):
            token_type = match.lastgroup
            token_value = match.group()
            
            # Skip whitespace
            if token_type == 'WHITESPACE':
                continue
                
            tokens.append(LaTeXToken(token_type, token_value, position))
            position += 1
        
        return tokens


class ASTNode:
    """Base class for Abstract Syntax Tree nodes"""
    pass


class NumberNode(ASTNode):
    def __init__(self, value: float):
        self.value = value


class VariableNode(ASTNode):
    def __init__(self, name: str):
        self.name = name


class BinaryOpNode(ASTNode):
    def __init__(self, left: ASTNode, operator: str, right: ASTNode):
        self.left = left
        self.operator = operator
        self.right = right


class UnaryOpNode(ASTNode):
    def __init__(self, operator: str, operand: ASTNode):
        self.operator = operator
        self.operand = operand


class FunctionNode(ASTNode):
    def __init__(self, function: str, args: List[ASTNode]):
        self.function = function
        self.args = args


class FractionNode(ASTNode):
    def __init__(self, numerator: ASTNode, denominator: ASTNode):
        self.numerator = numerator
        self.denominator = denominator


class LaTeXParser:
    """Parses LaTeX tokens into Abstract Syntax Tree"""
    
    def __init__(self, tokens: List[LaTeXToken]):
        self.tokens = tokens
        self.current = 0
        # Map Greek letters to their common names
        self.greek_map = {
            '\\alpha': 'alpha',
            '\\beta': 'beta',
            '\\gamma': 'gamma',
            '\\delta': 'delta',
            '\\sigma': 'sigma',
            '\\theta': 'theta',
            '\\lambda': 'lambda',
            '\\mu': 'mu',
            '\\pi': 'pi',
            '\\rho': 'rho'
        }
    
    def peek(self, offset: int = 0) -> LaTeXToken:
        """Look at token at current position + offset without consuming it"""
        pos = self.current + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return LaTeXToken('EOF', '', pos)
    
    def consume(self) -> LaTeXToken:
        """Consume and return current token"""
        token = self.peek()
        if self.current < len(self.tokens):
            self.current += 1
        return token
    
    def expect(self, token_type: str) -> LaTeXToken:
        """Consume token of expected type or raise error"""
        token = self.consume()
        if token.type != token_type:
            raise ValueError(f"Expected {token_type}, got {token.type}")
        return token
    
    def parse(self) -> ASTNode:
        """Parse tokens into AST"""
        # Skip leading dollar signs
        while self.peek().type == 'DOLLAR':
            self.consume()
        
        # Handle assignment (variable = expression)
        result = self.parse_assignment()
        
        # Skip trailing dollar signs
        while self.peek().type == 'DOLLAR':
            self.consume()
        
        return result
    
    def parse_assignment(self) -> ASTNode:
        """Parse assignment expression"""
        expr = self.parse_expression()
        
        # If we see an equals sign, this is an assignment
        if self.peek().type == 'EQUALS':
            self.consume()  # consume '='
            expr = self.parse_expression()  # return the RHS
        
        return expr
    
    def parse_expression(self) -> ASTNode:
        """Parse addition and subtraction (lowest precedence)"""
        left = self.parse_term()
        
        while self.peek().type in ['PLUS', 'MINUS']:
            op_token = self.consume()
            right = self.parse_term()
            left = BinaryOpNode(left, op_token.value, right)
        
        return left
    
    def parse_term(self) -> ASTNode:
        """Parse multiplication and division"""
        left = self.parse_power()
        
        while self.peek().type in ['MULTIPLY', 'DIVIDE', 'CDOT'] or self.is_implicit_multiplication():
            if self.is_implicit_multiplication():
                # Implicit multiplication (no operator between terms)
                op_value = '*'
            else:
                op_token = self.consume()
                op_value = '*' if op_token.value == '\\cdot' else op_token.value
            
            right = self.parse_power()
            left = BinaryOpNode(left, op_value, right)
        
        return left
    
    def is_implicit_multiplication(self) -> bool:
        """Check if next token implies multiplication"""
        token = self.peek()
        # Check for implicit multiplication but avoid treating subscripts as multiplication
        if self.peek().type == 'UNDERSCORE':
            return False
        return token.type in ['VARIABLE', 'LPAREN', 'NUMBER', 'FRAC', 'MAX', 'MIN', 'LOG', 
                              'ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 'THETA', 
                              'LAMBDA', 'MU', 'PI', 'RHO', 'VAR_BRACKET', 'TEXT']
    
    def parse_power(self) -> ASTNode:
        """Parse exponentiation (right associative)"""
        left = self.parse_factor()
        
        if self.peek().type == 'POWER':
            self.consume()  # consume '^'
            # Handle braced exponent
            if self.peek().type == 'LBRACE':
                self.consume()
                right = self.parse_expression()
                self.expect('RBRACE')
            else:
                right = self.parse_factor()
            return BinaryOpNode(left, '^', right)
        
        return left
    
    def parse_factor(self) -> ASTNode:
        """Parse primary expressions"""
        token = self.peek()
        
        if token.type == 'NUMBER':
            self.consume()
            return NumberNode(float(token.value))
        
        elif token.type == 'VAR_BRACKET':
            # Handle E[R_i] style variables
            self.consume()
            # Convert E[R_i] to E_R_i
            var_name = token.value.replace('[', '_').replace(']', '')
            return self.parse_subscript_continuation(VariableNode(var_name))
        
        elif token.type == 'VARIABLE':
            self.consume()
            var_name = token.value
            # Check for subscript
            if self.peek().type == 'UNDERSCORE':
                return self.parse_subscript(var_name)
            return VariableNode(var_name)
        
        elif token.type in ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 
                            'THETA', 'LAMBDA', 'MU', 'PI', 'RHO']:
            # Handle Greek letters
            self.consume()
            var_name = self.greek_map.get(token.value, token.value.strip('\\'))
            # Check for subscript
            if self.peek().type == 'UNDERSCORE':
                return self.parse_subscript(var_name)
            return VariableNode(var_name)
        
        elif token.type == 'E_CONST':
            self.consume()
            # Check if this is actually Euler's number or a variable E
            if self.peek().type == 'POWER':
                return NumberNode(math.e)
            else:
                # It's a variable E
                return VariableNode('e')
        
        elif token.type == 'TEXT':
            return self.parse_text_variable()
        
        elif token.type == 'FRAC':
            return self.parse_fraction()
        
        elif token.type in ['MAX', 'MIN']:
            return self.parse_minmax()
        
        elif token.type == 'LOG':
            return self.parse_log()
        
        elif token.type == 'LPAREN':
            self.consume()  # consume '('
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        
        elif token.type == 'LBRACE':
            self.consume()  # consume '{'
            expr = self.parse_expression()
            self.expect('RBRACE')
            return expr
        
        elif token.type == 'MINUS':
            self.consume()  # consume '-'
            operand = self.parse_power()  # Use parse_power to handle negative exponents properly
            return UnaryOpNode('-', operand)
        
        else:
            raise ValueError(f"Unexpected token: {token}")
    
    def parse_subscript(self, base_name: str) -> VariableNode:
        """Parse subscripted variable like R_f or beta_i"""
        self.consume()  # consume '_'
        
        # Handle braced subscript
        if self.peek().type == 'LBRACE':
            self.consume()
            subscript = ''
            while self.peek().type not in ['RBRACE', 'EOF']:
                token = self.consume()
                if token.type in self.greek_map:
                    subscript += self.greek_map[token.value]
                else:
                    subscript += token.value
            self.expect('RBRACE')
        # Handle Greek letter subscript
        elif self.peek().type in ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 
                                  'THETA', 'LAMBDA', 'MU', 'PI', 'RHO']:
            token = self.consume()
            subscript = self.greek_map.get(token.value, token.value.strip('\\'))
        # Handle single character subscript
        elif self.peek().type in ['VARIABLE', 'NUMBER']:
            token = self.consume()
            subscript = token.value
        else:
            raise ValueError(f"Expected subscript after underscore")
        
        return VariableNode(f"{base_name}_{subscript}")
    
    def parse_subscript_continuation(self, node: VariableNode) -> VariableNode:
        """Check if there's a subscript after a VAR_BRACKET variable"""
        if self.peek().type == 'UNDERSCORE':
            base_name = node.name
            self.consume()  # consume '_'
            
            # Handle braced subscript
            if self.peek().type == 'LBRACE':
                self.consume()
                subscript = ''
                while self.peek().type not in ['RBRACE', 'EOF']:
                    token = self.consume()
                    if token.type in self.greek_map:
                        subscript += self.greek_map[token.value]
                    else:
                        subscript += token.value
                self.expect('RBRACE')
            # Handle single character subscript
            elif self.peek().type in ['VARIABLE', 'NUMBER']:
                token = self.consume()
                subscript = token.value
            else:
                return node
            
            return VariableNode(f"{base_name}_{subscript}")
        return node
    
    def parse_text_variable(self) -> VariableNode:
        """Parse \\text{VariableName}"""
        self.expect('TEXT')
        self.expect('LBRACE')
        
        # Collect variable name (may contain underscores)
        var_name = ""
        while self.peek().type not in ['RBRACE', 'EOF']:
            token = self.consume()
            var_name += token.value
        
        self.expect('RBRACE')
        return VariableNode(var_name)
    
    def parse_fraction(self) -> FractionNode:
        """Parse \\frac{numerator}{denominator}"""
        self.expect('FRAC')
        self.expect('LBRACE')
        numerator = self.parse_expression()
        self.expect('RBRACE')
        self.expect('LBRACE')
        denominator = self.parse_expression()
        self.expect('RBRACE')
        return FractionNode(numerator, denominator)
    
    def parse_minmax(self) -> FunctionNode:
        """Parse \\max(...) or \\min(...)"""
        func_token = self.consume()  # MAX or MIN
        self.expect('LPAREN')
        
        args = []
        if self.peek().type != 'RPAREN':
            args.append(self.parse_expression())
            while self.peek().type == 'COMMA':
                self.consume()  # consume ','
                args.append(self.parse_expression())
        
        self.expect('RPAREN')
        return FunctionNode(func_token.value, args)
    
    def parse_log(self) -> FunctionNode:
        """Parse \\log(expression)"""
        self.expect('LOG')
        self.expect('LPAREN')
        arg = self.parse_expression()
        self.expect('RPAREN')
        return FunctionNode('\\log', [arg])


class LaTeXEvaluator:
    """Evaluates LaTeX AST with variable substitution"""
    
    def __init__(self, variables: Dict[str, float]):
        self.variables = variables
    
    def evaluate(self, node: ASTNode) -> float:
        """Evaluate AST node to numerical result"""
        if isinstance(node, NumberNode):
            return node.value
        
        elif isinstance(node, VariableNode):
            if node.name in self.variables:
                return self.variables[node.name]
            else:
                raise ValueError(f"Unknown variable: {node.name}")
        
        elif isinstance(node, BinaryOpNode):
            left_val = self.evaluate(node.left)
            right_val = self.evaluate(node.right)
            
            if node.operator == '+':
                return left_val + right_val
            elif node.operator == '-':
                return left_val - right_val
            elif node.operator == '*':
                return left_val * right_val
            elif node.operator == '/':
                if right_val == 0:
                    raise ValueError("Division by zero")
                return left_val / right_val
            elif node.operator == '^':
                return left_val ** right_val
            else:
                raise ValueError(f"Unknown binary operator: {node.operator}")
        
        elif isinstance(node, UnaryOpNode):
            operand_val = self.evaluate(node.operand)
            
            if node.operator == '-':
                return -operand_val
            else:
                raise ValueError(f"Unknown unary operator: {node.operator}")
        
        elif isinstance(node, FractionNode):
            numerator_val = self.evaluate(node.numerator)
            denominator_val = self.evaluate(node.denominator)
            
            if denominator_val == 0:
                raise ValueError("Division by zero in fraction")
            return numerator_val / denominator_val
        
        elif isinstance(node, FunctionNode):
            if node.function == '\\max':
                if not node.args:
                    raise ValueError("max() requires at least one argument")
                values = [self.evaluate(arg) for arg in node.args]
                return max(values)
            
            elif node.function == '\\min':
                if not node.args:
                    raise ValueError("min() requires at least one argument")
                values = [self.evaluate(arg) for arg in node.args]
                return min(values)
            
            elif node.function == '\\log':
                if len(node.args) != 1:
                    raise ValueError("log() requires exactly one argument")
                arg_val = self.evaluate(node.args[0])
                if arg_val <= 0:
                    raise ValueError("log() argument must be positive")
                return math.log(arg_val)
            
            else:
                raise ValueError(f"Unknown function: {node.function}")
        
        else:
            raise ValueError(f"Unknown node type: {type(node)}")


def evaluate_latex_formula(formula: str, variables: Dict[str, float]) -> float:
    """
    Main function to evaluate a LaTeX formula with given variables.
    
    Time Complexity: O(n) where n is formula length
    Space Complexity: O(n) for AST and recursion stack
    """
    try:
        # Tokenize the formula
        tokenizer = LaTeXTokenizer()
        tokens = tokenizer.tokenize(formula)
        
        # Parse tokens into AST
        parser = LaTeXParser(tokens)
        ast = parser.parse()
        
        # Evaluate AST with variables
        evaluator = LaTeXEvaluator(variables)
        result = evaluator.evaluate(ast)
        
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating formula '{formula}': {str(e)}")
        raise


@latex_bp.route('/trading-formula', methods=['POST'])
def trading_formula():
    """
    POST endpoint for LaTeX formula evaluation.
    
    Expected input: JSON array of test cases
    Expected output: JSON array of results
    """
    try:
        # Validate content type
        if request.content_type != 'application/json':
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Get input data
        input_data = request.get_json()
        
        if not isinstance(input_data, list):
            return jsonify({'error': 'Expected JSON array'}), 400
        
        results = []
        
        for test_case in input_data:
            try:
                # Validate test case structure
                if not isinstance(test_case, dict):
                    raise ValueError("Test case must be an object")
                
                if 'formula' not in test_case or 'variables' not in test_case:
                    raise ValueError("Test case must have 'formula' and 'variables' fields")
                
                formula = test_case['formula']
                variables = test_case['variables']
                
                # Evaluate the formula
                result = evaluate_latex_formula(formula, variables)
                
                # Round to 4 decimal places as required
                rounded_result = round(result, 4)
                
                results.append({"result": rounded_result})
                
            except Exception as e:
                logger.error(f"Error processing test case: {str(e)}")
                # Return error for this specific test case
                results.append({"error": str(e)})
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in trading-formula endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500