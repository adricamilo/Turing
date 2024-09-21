from typing import Union, List, Tuple, Generator
import copy

class AbstractSymbol:
    """
    Base class for symbols used in the Turing Machine.
    """
    _subscript_map = {
            '0': '\u2080', '1': '\u2081', '2': '\u2082', '3': '\u2083', '4': '\u2084',
            '5': '\u2085', '6': '\u2086', '7': '\u2087', '8': '\u2088', '9': '\u2089'
        }

    def int_to_subscript(self, n: int) -> str:
        """Convert an integer to its subscript form."""
        return ''.join(self._subscript_map[digit] for digit in str(n))

class InternalConfig(AbstractSymbol):
    """
    Represents an internal configuration (state) of the Turing Machine.
    """
    def __init__(self, config: int):
        if config < 1:
            raise ValueError("Invalid internal config. Only positive integers allowed.") 
        self.config = config

    def __eq__(self, other: 'InternalConfig') -> bool:
        return isinstance(other, InternalConfig) and self.config == other.config

    def __str__(self) -> str:
        return f'q{self.int_to_subscript(self.config)}'

    def __repr__(self) -> str:
        return str(self)

class AlphabetSymbol(AbstractSymbol):
    """
    Represents an alphabet symbol used in the Turing Machine.
    """
    def __init__(self, symbol: int):
        if symbol < 0:
            raise ValueError("Invalid symbol. Only non-negative integers allowed.") 
        self.symbol = symbol
        
    def __eq__(self, other: 'AlphabetSymbol') -> bool:
        return isinstance(other, AlphabetSymbol) and self.symbol == other.symbol

    def __str__(self) -> str:
        if self.symbol == 0:
            return 'B'  # Blank symbol
        elif self.symbol == 1:
            return '1'
        return f"S{self.int_to_subscript(self.symbol)}"

    def __repr__(self) -> str:
        return str(self)

class Quadruple:
    """
    Represents a quadruple (instruction) of the Turing Machine.
    """
    def __init__(self, e1: int, e2: int, e3: str, e4: int):
        self.e1 = InternalConfig(e1)
        self.e2 = AlphabetSymbol(e2)
        self.e4 = InternalConfig(e4)
        self.e3 = self._parse_third_element(str(e3))

    def _parse_third_element(self, e3: str) -> Union[str, AbstractSymbol]:
        if e3 in ('R', 'L'):
            return e3
        elif e3 == 'B':
            return AlphabetSymbol(0)
        elif e3 == '1':
            return AlphabetSymbol(1)
        if e3.startswith('S'):
            return AlphabetSymbol(self._parse_numeric(e3[1:], e3))
        if e3.startswith('q'):
            return InternalConfig(self._parse_numeric(e3[1:], e3))
        raise ValueError(f"'{e3}' is an invalid quadruple third element.")

    def _parse_numeric(self, value: str, element: str) -> int:
        """Helper method to parse some numeric values."""
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"'{element}' is an invalid quadruple third element.")
    
    def __getitem__(self, key: int) -> Union[str, AbstractSymbol]:
        return [self.e1, self.e2, self.e3, self.e4][key]

    def __iter__(self) -> Generator[Union[AbstractSymbol, str], None, None]:
        yield from [self.e1, self.e2, self.e3, self.e4]

    def __eq__(self, other: 'Quadruple') -> bool:
        return isinstance(other, Quadruple) and self.e1 == other.e1 and self.e2 == other.e2
    
    def __str__(self) -> str:
        return f"[{self.e1} {self.e2} {self.e3} {self.e4}]"

    def __repr__(self) -> str:
        return str(self)

class InstantaneousDescription:
    """
    Represents the instantaneous description (current state) of the Turing Machine.
    """
    def __init__(self, symbol_sequence: List[AbstractSymbol]):
        if not all(isinstance(symbol, AbstractSymbol) for symbol in symbol_sequence):
            raise ValueError("Description must only contain symbols.")
        if sum(isinstance(symbol, InternalConfig) for symbol in symbol_sequence) != 1:
            raise ValueError("Description must contain exactly one internal config.")
        if isinstance(symbol_sequence[-1], InternalConfig):
            raise ValueError("Internal config must not be the last element in the description.")

        self.symbol_sequence = symbol_sequence

    def __str__(self) -> str:
        return ''.join(str(symbol) for symbol in self.symbol_sequence)

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: int) -> AbstractSymbol:
        return self.symbol_sequence[key]

    def __len__(self) -> int:
        return len(self.symbol_sequence)

    def internal_config_index(self) -> int:
        """Return the index of the internal configuration at this description."""
        for i, symbol in enumerate(self.symbol_sequence):
            if isinstance(symbol, InternalConfig):
                return i
        raise ValueError("Internal config not found in description.")

    def internal_config(self) -> InternalConfig:
        """Return the internal configuration at the description."""
        index = self.internal_config_index()
        return self.symbol_sequence[index]

    def scanned_symbol(self) -> AlphabetSymbol:
        """Return the symbol being scanned (next to the internal configuration)."""
        index = self.internal_config_index() + 1
        return self.symbol_sequence[index]

    def count_ones(self) -> int:
        """Counts the number of AlphabetSymbol(1)."""
        return sum(isinstance(symbol, AlphabetSymbol) and symbol.symbol == 1 for symbol in self.symbol_sequence)

class TuringMachine:
    """
    Represents a Turing Machine with a set of quadruples (instructions).
    """
    def __init__(self, quadruples: List[Quadruple], max_computations: int = 10000):
        if not all(isinstance(quadruple, Quadruple) for quadruple in quadruples):
            raise ValueError("All list elements must be quadruples.")

        if not isinstance(max_computations, int) or max_computations < 0:
            raise ValueError("The maximum number of computations must be a non-negative integer.")
        self.max_computations = max_computations
        
        self.quadruples = []
        for quadruple in quadruples:
            # Check for duplicate quadruples
            if quadruple in self.quadruples:
                raise ValueError("Machine cannot contain two quadruples with the same first two elements.")
            self.quadruples.append(quadruple)

        # Add a memoization dictionary
        self._computation_cache = {}
        
    def __getitem__(self, key: int) -> Quadruple:
        return self.quadruples[key]

    def __len__(self) -> int:
        return len(self.quadruples)

    def __iter__(self) -> Generator[Quadruple, None, None]:
        yield from self.quadruples
    
    def __str__(self) -> str:
        return f"[{self.quadruples}, max_computations={self.max_computations}]"

    def __repr__(self) -> str:
        return str(self)

    def _input_form(self, args: List[int]) -> List[AbstractSymbol]:
        description = [InternalConfig(1)]
        for arg in args[:-1]:
            for i in range(arg + 1):
                description.append(AlphabetSymbol(1))
            description.append(AlphabetSymbol(0))
        for i in range(args[-1] + 1):
            description.append(AlphabetSymbol(1))

        return description

    def _next_compute_index(self, internal_config: InternalConfig, scanned_symbol: AlphabetSymbol) -> int:
        """
        Find the index of the next quadruple to execute based on the current internal configuration and scanned symbol.
        Returns -1 if no matching quadruple is found.
        """
        for i, quadruple in enumerate(self.quadruples):
            if quadruple[0] == internal_config and quadruple[1] == scanned_symbol:
                return i
        return -1

    def _process_instruction(self, instruction: Quadruple, description: InstantaneousDescription) -> InstantaneousDescription:
        """
        Process a single instruction (quadruple) and return the resulting instantaneous description.
        """
        new_description_list = copy.deepcopy(description.symbol_sequence)
        internal_config_index = description.internal_config_index()
        
        if isinstance(instruction[2], AlphabetSymbol):
            # Write symbol and update internal config
            new_description_list[internal_config_index + 1] = copy.copy(instruction[2])
            new_description_list[internal_config_index] = copy.copy(instruction[3])
        elif isinstance(instruction[2], InternalConfig):
            # See Sec. 4
            raise Exception("Unsupported operation.")
        elif instruction[2] == "R":
            # Move right
            new_description_list[internal_config_index] = new_description_list[internal_config_index + 1]
            new_description_list[internal_config_index + 1] = copy.copy(instruction[3])
            if internal_config_index == len(description) - 2:
                new_description_list.append(AlphabetSymbol(0))  # Extend tape with blank symbol if at the end
        elif instruction[2] == "L":
            # Move left
            if internal_config_index == 0:
                new_description_list.insert(0, AlphabetSymbol(0))  # Extend tape with blank symbol if at the beginning
                internal_config_index += 1
            new_description_list[internal_config_index] = new_description_list[internal_config_index - 1]
            new_description_list[internal_config_index - 1] = copy.copy(instruction[3])

        return InstantaneousDescription(new_description_list)
    
    def compute(self, args: List[int]) -> List[Tuple[InstantaneousDescription, Union[Quadruple, None]]]:
        """
        Perform the Turing machine computation for the given input arguments.
        Returns a list of tuples, where each tuple contains an instantaneous description and the quadruple used to get there.
        Uses memoization to avoid repeating computations.
        """
        if not all(isinstance(arg, int) and arg >= 0 for arg in args):
            raise ValueError("Arguments must be positive integers.")

        # Convert args to a hashable tuple for dictionary key
        args_key = tuple(args)

        # Check if the computation result is already in the cache
        if args_key in self._computation_cache:
            return self._computation_cache[args_key]
        
        # Initialize computation with initial input
        computation = [(InstantaneousDescription(self._input_form(args)), None)]

        # Start the computation loop
        number_of_computations = 1
        while True:
            current_description = computation[-1][0]
            current_internal_config = current_description.internal_config()
            current_scanned_symbol = current_description.scanned_symbol()

            next_compute_index = self._next_compute_index(current_internal_config, current_scanned_symbol)
            if next_compute_index == -1:
                break  # No matching quadruple, halt the machine
            
            current_instruction = self.quadruples[next_compute_index]
            new_description = self._process_instruction(current_instruction, current_description)
            
            computation.append((new_description, current_instruction))
            
            # Check if maximum number of computations is reached
            number_of_computations += 1
            if number_of_computations > self.max_computations:
                break
            
        # Cache the computation result before returning
        self._computation_cache[args_key] = computation
        
        return computation

    def clear_cache(self):
        self._computation_cache.clear()

    def resultant(self, args: List[int]) -> InstantaneousDescription:
        return self.compute(args)[-1][0]
