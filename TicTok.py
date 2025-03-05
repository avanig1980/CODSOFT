import random
import math
import logging
from enum import Enum, auto

class GameMode(Enum):
    MINIMAX = auto()
    STRATEGIC = auto()
    MACHINE_LEARNING = auto()

class TicTacToeAI:
    def __init__(self, mode=GameMode.MINIMAX, difficulty='hard', log_file='tictactoe.log'):
        # Initialize logging
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Board initialization
        self.board = [' ' for _ in range(9)]
        self.winning_combinations = [
            # Rows
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            # Columns
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            # Diagonals
            [0, 4, 8], [2, 4, 6]
        ]
        
        # Game mode and difficulty settings
        self.mode = mode
        self.difficulty = difficulty
        
        # Machine Learning Q-learning parameters
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        
        # Strategic move weights
        self.strategy_weights = {
            'center_control': 1.5,
            'corner_control': 1.2,
            'blocking': 1.3,
            'winning_move': 2.0
        }
    
    def print_board(self):
        """Print the current state of the board"""
        for i in range(0, 9, 3):
            print(f' {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} ')
            if i < 6:
                print('-----------')
    
    def is_winner(self, player):
        """Check if the given player has won"""
        for combo in self.winning_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_board_full(self):
        """Check if the board is completely filled"""
        return ' ' not in self.board
    
    def get_empty_squares(self):
        """Return a list of empty square indices"""
        return [i for i, square in enumerate(self.board) if square == ' ']
    
    def make_move(self, position, player):
        """Place a player's mark on the board"""
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False
    
    def minimax(self, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
        """
        Minimax algorithm with Alpha-Beta Pruning
        
        Args:
            depth (int): Current depth of the search tree
            is_maximizing (bool): Whether it's the maximizing player's turn
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
        
        Returns:
            int: Best score for the current board state
        """
        # Check terminal states
        if self.is_winner('X'):
            return -10 + depth
        if self.is_winner('O'):
            return 10 - depth
        if self.is_board_full():
            return 0
        
        if is_maximizing:
            best_score = -math.inf
            for pos in self.get_empty_squares():
                # Try the move
                self.board[pos] = 'O'
                # Recursively evaluate
                score = self.minimax(depth + 1, False, alpha, beta)
                # Undo the move
                self.board[pos] = ' '
                
                # Update best score and alpha
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                
                # Prune the search tree
                if beta <= alpha:
                    break
            return best_score
        
        else:
            best_score = math.inf
            for pos in self.get_empty_squares():
                # Try the move
                self.board[pos] = 'X'
                # Recursively evaluate
                score = self.minimax(depth + 1, True, alpha, beta)
                # Undo the move
                self.board[pos] = ' '
                
                # Update best score and beta
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                
                # Prune the search tree
                if beta <= alpha:
                    break
            return best_score
    
    def evaluate_move_strategically(self, position):
        """
        Evaluate a move based on strategic considerations
        
        Args:
            position (int): Board position to evaluate
        
        Returns:
            float: Strategic move score
        """
        strategic_score = 1.0  # Base score
        
        # Center control is most strategic
        if position == 4:
            strategic_score *= self.strategy_weights['center_control']
        
        # Corner control is next most important
        if position in [0, 2, 6, 8]:
            strategic_score *= self.strategy_weights['corner_control']
        
        # Check for blocking potential
        if self.would_block_opponent(position):
            strategic_score *= self.strategy_weights['blocking']
        
        # Check for immediate winning move
        if self.would_win(position):
            strategic_score *= self.strategy_weights['winning_move']
        
        return strategic_score
    
    def would_block_opponent(self, position):
        """
        Check if a move would block the opponent's potential win
        
        Args:
            position (int): Position to check
        
        Returns:
            bool: Whether the move blocks a potential winning line
        """
        # Temporarily place the move
        self.board[position] = 'O'
        
        # Check if this prevents opponent's win
        for combo in self.winning_combinations:
            if all(self.board[i] == 'O' or self.board[i] == ' ' for i in combo):
                blocked_lines = sum(
                    1 for i in combo 
                    if self.board[i] == 'X'
                )
                # Reset the board
                self.board[position] = ' '
                return blocked_lines >= 2
        
        # Reset the board
        self.board[position] = ' '
        return False
    
    def would_win(self, position):
        """
        Check if a move results in an immediate win
        
        Args:
            position (int): Position to check
        
        Returns:
            bool: Whether the move creates a winning scenario
        """
        # Temporarily place the move
        self.board[position] = 'O'
        is_winning = self.is_winner('O')
        # Reset the board
        self.board[position] = ' '
        return is_winning
    
    def state_to_key(self):
        """
        Convert board state to hashable key
        
        Returns:
            tuple: Immutable board state representation
        """
        return tuple(self.board)
    
    def get_best_move(self):
        """
        Select the best move based on game mode
        
        Returns:
            int: Best move index
        """
        if self.mode == GameMode.MINIMAX:
            return self._minimax_best_move()
        elif self.mode == GameMode.STRATEGIC:
            return self._strategic_best_move()
        elif self.mode == GameMode.MACHINE_LEARNING:
            return self._machine_learning_best_move()
    
    def _minimax_best_move(self):
        """
        Find the best move using Minimax algorithm
        
        Returns:
            int: Best move index
        """
        best_score = -math.inf
        best_move = None
        
        # Try all empty squares
        for pos in self.get_empty_squares():
            # Try the move
            self.board[pos] = 'O'
            # Calculate the move's score
            score = self.minimax(0, False)
            # Undo the move
            self.board[pos] = ' '
            
            # Update best move if found a better score
            if score > best_score:
                best_score = score
                best_move = pos
        
        return best_move
    
    def _strategic_best_move(self):
        """
        Enhanced best move selection with strategic evaluation
        
        Returns:
            int: Best move index
        """
        best_strategic_moves = []
        best_strategic_score = -math.inf
        
        for pos in self.get_empty_squares():
            # Try the move
            self.board[pos] = 'O'
            # Calculate minimax score
            minimax_score = self.minimax(0, False)
            # Calculate strategic score
            strategic_score = self.evaluate_move_strategically(pos)
            
            # Combine scores
            combined_score = minimax_score * strategic_score
            
            # Undo the move
            self.board[pos] = ' '
            
            # Track best moves
            if combined_score > best_strategic_score:
                best_strategic_moves = [pos]
                best_strategic_score = combined_score
            elif combined_score == best_strategic_score:
                best_strategic_moves.append(pos)
        
        # Randomly choose among equally good moves
        return random.choice(best_strategic_moves)
    
    def _machine_learning_best_move(self):
        """
        Q-learning based move selection
        
        Returns:
            int: Best move index
        """
        state_key = self.state_to_key()
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            return random.choice(self.get_empty_squares())
        
        # Initialize Q-values for this state if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {pos: 0 for pos in self.get_empty_squares()}
        
        # Select move with highest Q-value
        return max(
            self.q_table[state_key], 
            key=self.q_table[state_key].get
        )
    
    def update_q_learning(self, state, move, reward):
        """
        Update Q-table based on game outcome
        
        Args:
            state (tuple): Board state
            move (int): Selected move
            reward (float): Game outcome reward
        """
        if state not in self.q_table:
            self.q_table[state] = {move: 0}
        
        # Q-learning update rule
        current_q = self.q_table[state].get(move, 0)
        self.q_table[state][move] = current_q + self.learning_rate * (
            reward - current_q
        )
    
    def get_difficulty_move(self):
        """
        AI move selection with varying difficulty
        
        Returns:
            int: Selected move index
        """
        if self.difficulty == 'easy':
            # Random move selection
            return random.choice(self.get_empty_squares())
        
        elif self.difficulty == 'medium':
            # 70% chance of optimal move, 30% random
            if random.random() < 0.7:
                return self.get_best_move()
            else:
                return random.choice(self.get_empty_squares())
        
        else:  # hard mode
            return self.get_best_move()
    
    def play(self):
        """Main game loop"""
        print("Welcome to Advanced Tic-Tac-Toe!")
        print(f"Game Mode: {self.mode.name}")
        print(f"Difficulty: {self.difficulty}")
        print("You are 'X', the AI is 'O'")
        print("Enter a number 0-8 corresponding to the board positions:")
        print(" 0 | 1 | 2 ")
        print("-----------")
        print(" 3 | 4 | 5 ")
        print("-----------")
        print(" 6 | 7 | 8 ")
        
        # Randomly decide who goes first
        current_player = random.choice(['X', 'O'])
        
        # Logging game start
        self.logger.info(f"New Game Started - Mode: {self.mode.name}, Difficulty: {self.difficulty}")
        
        while True:
            self.print_board()
            
            if current_player == 'X':
                # Human player's turn
                while True:
                    try:
                        move = int(input("Enter your move (0-8): "))
                        if move in self.get_empty_squares():
                            self.make_move(move, 'X')
                            break
                        else:
                            print("That square is already occupied. Try again.")
                    except (ValueError, IndexError):
                        print("Invalid input. Enter a number between 0 and 8.")
            else:
                # AI's turn
                if self.difficulty in ['easy', 'medium', 'hard']:
                    move = self.get_difficulty_move()
                else:
                    move = self.get_best_move()
                
                print(f"AI chooses position {move}")
                self.make_move(move, 'O')
                
                # Log AI move
                self.logger.info(f"AI Move: {move}")
            
            # Check for win or draw
            if self.is_winner('X'):
                self.print_board()
                print("Congratulations! You won!")
                self.logger.info("Game Result: Human Won")
                break
            elif self.is_winner('O'):
                self.print_board()
                print("AI wins! Better luck next time.")
                self.logger.info("Game Result: AI Won")
                break
            elif self.is_board_full():
                self.print_board()
                print("It's a draw!")
                self.logger.info("Game Result: Draw")
                break
            
            # Switch players
            current_player = 'O' if current_player == 'X' else 'X'

# Run the game with different modes and difficulties
def main():
    print("Select Game Mode:")
    print("1. Minimax AI")
    print("2. Strategic AI")
    print("3. Machine Learning AI")
    
    mode_choice = input("Enter mode number (1-3): ")
    mode_map = {
        '1': GameMode.MINIMAX,
        '2': GameMode.STRATEGIC,
        '3': GameMode.MACHINE_LEARNING
    }
    
    print("\nSelect Difficulty:")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")
    
    difficulty_choice = input("Enter difficulty number (1-3): ")
    difficulty_map = {
        '1': 'easy',
        '2': 'medium',
        '3': 'hard'
    }
    
    game = TicTacToeAI(
        mode=mode_map.get(mode_choice, GameMode.MINIMAX),
        difficulty=difficulty_map.get(difficulty_choice, 'hard')
    )
    game.play()

if __name__ == "__main__":
    main()