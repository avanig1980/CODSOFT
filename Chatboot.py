import re
import random
import logging
import traceback
from datetime import datetime

class AdvancedRuleBasedChatbot:
    def __init__(self, log_file='chatbot.log'):
        # Logging Configuration
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Context Tracking
        self.context = {
            'last_topic': None,
            'conversation_history': [],
            'user_name': None,
            'start_time': datetime.now()
        }
        
        # Predefined Response Patterns
        self.patterns = {
            'greetings': [
                (r'\b(hi|hello|hey|greeting)\b', [
                    "Hello there! How can I help you today?",
                    "Hi! What can I do for you?",
                    "Greetings! How are you doing?"
                ], 0.7),
                (r'\b(how are you|how do you do)\b', [
                    "I'm doing well, thank you for asking!",
                    "I'm great! How about you?",
                    "Perfectly fine, ready to assist you!"
                ], 0.8)
            ],
            
            'farewells': [
                (r'\b(bye|goodbye|exit|quit)\b', [
                    "Goodbye! Have a great day!",
                    "See you later! Take care.",
                    "Bye! Hope I could help you."
                ], 0.9)
            ],
            
            'emotions': [
                (r'\b(angry|upset|frustrated)\b', [
                    "I sense you're feeling frustrated. Would you like to talk about it?",
                    "It sounds like something is bothering you. How can I help?"
                ], 0.8),
                (r'\b(happy|excited|great)\b', [
                    "That's wonderful! I'm glad you're feeling positive.",
                    "It's great to hear you're in a good mood!"
                ], 0.7)
            ],
            
            'help_queries': [
                (r'\b(help|support|assist)\b', [
                    "I'm here to help! What do you need assistance with?",
                    "Sure, I can help. What seems to be the problem?",
                    "Support is my specialty. What can I do for you?"
                ], 0.6)
            ],
            
            'knowledge_base': [
                (r'\b(who are you|what can you do)\b', [
                    "I'm an advanced rule-based chatbot designed to understand and respond to various queries.",
                    "I use sophisticated pattern matching and context tracking to provide helpful responses.",
                    "I can help with conversation, answer questions, and track context."
                ], 0.7)
            ]
        }
    
    def match_intent(self, message):
        """
        Advanced intent matching with weighted scoring
        
        Args:
            message (str): User input
        
        Returns:
            dict: Matched intent details
        """
        message = message.lower()
        best_match = {
            'intent': None,
            'confidence': 0,
            'response': None
        }
        
        # Check all pattern categories
        for category, pattern_list in self.patterns.items():
            for pattern, responses, weight in pattern_list:
                match = re.search(pattern, message)
                if match:
                    # Calculate confidence based on match and weight
                    confidence = weight * len(match.group(0)) / len(message)
                    
                    if confidence > best_match['confidence']:
                        best_match = {
                            'intent': category,
                            'confidence': confidence,
                            'response': random.choice(responses)
                        }
        
        return best_match
    
    def update_context(self, user_input, response):
        """
        Update conversation context
        
        Args:
            user_input (str): User's most recent message
            response (str): Chatbot's response
        """
        # Detect and store potential name
        name_match = re.search(r'my name is (\w+)', user_input.lower())
        if name_match:
            self.context['user_name'] = name_match.group(1).capitalize()
        
        # Store conversation history
        self.context['conversation_history'].append({
            'user': user_input,
            'bot': response
        })
        
        # Limit history to last 5 interactions
        if len(self.context['conversation_history']) > 5:
            self.context['conversation_history'].pop(0)
    
    def personalize_response(self, response):
        """
        Add personalization to responses
        
        Args:
            response (str): Original response
        
        Returns:
            str: Personalized response
        """
        if self.context['user_name']:
            personalized_responses = [
                f"{self.context['user_name']}, {response}",
                f"Sure, {self.context['user_name']}! {response}"
            ]
            return random.choice(personalized_responses)
        return response
    
    def log_conversation(self, user_input, bot_response):
        """
        Log conversation details
        
        Args:
            user_input (str): User's message
            bot_response (str): Bot's response
        """
        try:
            self.logger.info(f"User Input: {user_input}")
            self.logger.info(f"Bot Response: {bot_response}")
        except Exception as e:
            self.logger.error(f"Logging error: {e}")
            self.logger.error(traceback.format_exc())
    
    def generate_response(self, user_input):
        """
        Generate a contextual response
        
        Args:
            user_input (str): User's input message
        
        Returns:
            str: Chatbot's response
        """
        # Match intent
        intent_match = self.match_intent(user_input)
        
        # Generate response
        if intent_match['response']:
            response = intent_match['response']
        else:
            # Default responses for unmatched intents
            default_responses = [
                "I'm not sure how to respond to that.",
                "Could you rephrase that?",
                "I don't understand. Can you be more specific?"
            ]
            response = random.choice(default_responses)
        
        # Personalize response
        response = self.personalize_response(response)
        
        # Update context
        self.update_context(user_input, response)
        
        # Log conversation
        self.log_conversation(user_input, response)
        
        return response
    
    def chat(self):
        """
        Start an interactive chat session
        """
        print("Advanced Rule-Based Chatbot: Hello! Type 'quit' or 'exit' to end the conversation.")
        
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
                
                # Check for exit conditions
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Advanced Rule-Based Chatbot: Goodbye!")
                    break
                
                # Generate and print response
                response = self.generate_response(user_input)
                print("Advanced Rule-Based Chatbot:", response)
            
            except Exception as e:
                print("An error occurred. Please try again.")
                self.logger.error(f"Chat session error: {e}")
                self.logger.error(traceback.format_exc())
        
        # Calculate conversation duration
        end_time = datetime.now()
        duration = end_time - self.context['start_time']
        self.logger.info(f"Conversation duration: {duration}")

# Run the chatbot
if __name__ == "__main__":
    chatbot = AdvancedRuleBasedChatbot()
    chatbot.chat()