import json
import re
import yaml
import random
import os
import logging
from typing import List, Dict, Any, Tuple
from datasets import Dataset
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVChargingQAGenerator:
    """Generate QA pairs for EV charging domain from collected data."""
    
    def __init__(self):
        # Define question templates for different EV charging topics
        self.question_templates = {
            'connector_types': [
                "What are the different types of EV charging connectors?",
                "What is the difference between Type 1 and Type 2 EV connectors?",
                "Which charging connector standards are used in different regions?",
                "What is CHAdeMO and how does it compare to CCS?",
                "What connector types does Tesla use for charging?",
                "How do charging connector standards affect EV interoperability?"
            ],
            'charging_levels': [
                "What are the different levels of EV charging?",
                "How long does it take to charge an electric vehicle?",
                "What is the difference between AC and DC charging?",
                "What factors affect EV charging time?",
                "What is fast charging and how does it work?",
                "What are the power levels for Level 1, 2, and 3 charging?"
            ],
            'infrastructure': [
                "Where should EV charging stations be located?",
                "What are the key considerations for charging infrastructure planning?",
                "How does workplace charging differ from public charging?",
                "What are the costs associated with installing EV charging stations?",
                "How does charging station utilization vary by location?",
                "What technical requirements are needed for EV charging infrastructure?"
            ],
            'smart_charging': [
                "What is smart charging and how does it work?",
                "How can EVs integrate with the electrical grid?",
                "What is Vehicle-to-Grid (V2G) technology?",
                "How does time-of-use pricing affect EV charging?",
                "What role do EVs play in renewable energy integration?",
                "How does load balancing work with multiple EVs charging?"
            ],
            'user_behavior': [
                "How do EV users typically charge their vehicles?",
                "What factors influence EV charging behavior?",
                "How does weather affect EV charging patterns?",
                "What are the differences between residential and public charging usage?",
                "How do early adopters differ from late adopters in charging behavior?",
                "What pricing models are used for EV charging services?"
            ],
            'technology_trends': [
                "What are the latest trends in EV charging technology?",
                "How is EV charging infrastructure evolving?",
                "What innovations are improving EV charging speed?",
                "How are charging networks expanding globally?",
                "What role does artificial intelligence play in EV charging?",
                "How is wireless charging technology developing for EVs?"
            ]
        }
        
        # Define topic keywords to match content with question categories
        self.topic_keywords = {
            'connector_types': ['connector', 'plug', 'chademo', 'ccs', 'j1772', 'type 1', 'type 2', 'tesla', 'supercharger'],
            'charging_levels': ['level 1', 'level 2', 'level 3', 'ac charging', 'dc charging', 'fast charging', 'power', 'kw'],
            'infrastructure': ['station', 'infrastructure', 'deployment', 'planning', 'location', 'workplace', 'public', 'cost'],
            'smart_charging': ['smart charging', 'grid', 'v2g', 'vehicle-to-grid', 'load balancing', 'renewable', 'demand response'],
            'user_behavior': ['behavior', 'usage', 'pattern', 'user', 'adoption', 'pricing', 'utilization', 'session'],
            'technology_trends': ['trend', 'innovation', 'future', 'technology', 'development', 'network', 'wireless']
        }

    def categorize_content(self, text: str) -> str:
        """Categorize content based on keywords to match with appropriate questions."""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        # Return topic with highest score, default to 'infrastructure' if no clear match
        best_topic = max(topic_scores.items(), key=lambda x: x[1])
        return best_topic[0] if best_topic[1] > 0 else 'infrastructure'

    def extract_relevant_content(self, text: str, max_length: int = 500) -> str:
        """Extract the most relevant portion of text for answering questions."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Remove very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Build answer by adding sentences until we reach max_length
        answer_parts = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            answer_parts.append(sentence)
            current_length += len(sentence)
        
        return '. '.join(answer_parts).strip() + '.'

    def generate_contextual_questions(self, text: str, topic: str) -> List[str]:
        """Generate contextual questions based on text content."""
        base_questions = self.question_templates.get(topic, self.question_templates['infrastructure'])
        
        # Add some content-specific questions
        contextual_questions = []
        text_lower = text.lower()
        
        # Generate questions based on specific content
        if 'station' in text_lower and 'location' in text_lower:
            contextual_questions.append("What factors should be considered when selecting locations for EV charging stations?")
        
        if 'cost' in text_lower or 'price' in text_lower:
            contextual_questions.append("What are the cost considerations for EV charging?")
        
        if 'time' in text_lower and 'charging' in text_lower:
            contextual_questions.append("How does charging time vary for different types of EV charging?")
        
        if 'battery' in text_lower:
            contextual_questions.append("How do battery characteristics affect EV charging?")
        
        # Combine base questions with contextual ones
        all_questions = base_questions + contextual_questions
        return random.sample(all_questions, min(3, len(all_questions)))

    def create_llama3_format(self, question: str, answer: str, system_prompt: str = None) -> Dict[str, str]:
        """Format QA pair for Llama 3 training."""
        if system_prompt is None:
            system_prompt = "You are an expert on electric vehicle charging technology and infrastructure. Provide accurate, detailed, and helpful information about EV charging."
        
        # Llama 3 chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # Create instruction format for fine-tuning
        instruction_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        
        return {
            "messages": messages,
            "text": instruction_text,
            "question": question,
            "answer": answer,
            "system_prompt": system_prompt
        }

    def generate_qa_pairs(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate QA pairs from processed data."""
        qa_pairs = []
        
        logger.info(f"Generating QA pairs from {len(processed_data)} data items")
        
        for item in processed_data:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            
            # Skip very short texts
            if len(text) < 100:
                continue
            
            # Categorize content
            topic = self.categorize_content(text)
            
            # Generate questions for this content
            questions = self.generate_contextual_questions(text, topic)
            
            # Extract relevant content for answer
            answer_content = self.extract_relevant_content(text)
            
            # Create QA pairs
            for question in questions:
                qa_pair = self.create_llama3_format(question, answer_content)
                qa_pair['metadata'] = {
                    'source': metadata.get('source', 'unknown'),
                    'topic': topic,
                    'data_type': metadata.get('type', 'unknown'),
                    'chunk_id': metadata.get('chunk_id', 0)
                }
                qa_pairs.append(qa_pair)
        
        logger.info(f"Generated {len(qa_pairs)} QA pairs")
        return qa_pairs

    def create_specialized_qa_pairs(self) -> List[Dict[str, str]]:
        """Create additional specialized QA pairs for comprehensive coverage."""
        specialized_pairs = []
        
        # Define comprehensive QA pairs for key EV charging concepts
        comprehensive_qa = [
            {
                "question": "What are the main types of EV charging connectors and their characteristics?",
                "answer": "The main EV charging connector types include: Type 1 (SAE J1772) used in North America and Japan for AC charging up to 7.4kW; Type 2 (IEC 62196) European standard for AC charging up to 22kW; CHAdeMO Japanese DC fast charging standard up to 50kW+; CCS (Combined Charging System) that combines AC and DC charging in one connector; Tesla Supercharger proprietary standard for high-speed DC charging; and GB/T Chinese national standard for both AC and DC charging.",
                "topic": "connector_types"
            },
            {
                "question": "How do the three levels of EV charging differ in terms of power and charging time?",
                "answer": "Level 1 charging uses 120V household outlets providing 1.4kW power, requiring 8-20 hours for a full charge. Level 2 charging uses 240V providing 3.7-22kW power, requiring 3-8 hours for a full charge. Level 3 charging is DC fast charging providing 50-350kW power, requiring only 20-60 minutes for 80% charge. Charging time depends on battery capacity, current state of charge, vehicle's maximum charging rate, ambient temperature, and battery management system limitations.",
                "topic": "charging_levels"
            },
            {
                "question": "What factors should be considered when planning EV charging infrastructure deployment?",
                "answer": "EV charging infrastructure planning requires considering location factors like high traffic areas, workplace and residential needs, highway corridors, and shopping centers. Technical requirements include electrical grid capacity, power distribution, transformer sizing, network connectivity, and ADA compliance. Economic considerations involve capital costs, operational and maintenance costs, revenue models, pricing strategies, and government incentives. Proper planning ensures adequate coverage while maximizing utilization and return on investment.",
                "topic": "infrastructure"
            },
            {
                "question": "How does smart charging technology optimize EV charging and grid integration?",
                "answer": "Smart charging optimizes EV charging through time-of-use pricing to reduce costs and grid stress during off-peak hours, load balancing to distribute charging across multiple vehicles preventing grid overload, Vehicle-to-Grid (V2G) technology allowing EVs to supply power back to the grid during peak demand, renewable integration by charging when solar/wind generation is high, and demand response by adjusting charging rates based on grid conditions. This requires communication between vehicles, chargers, and grid operators for optimal energy management.",
                "topic": "smart_charging"
            }
        ]
        
        for qa in comprehensive_qa:
            formatted_qa = self.create_llama3_format(qa["question"], qa["answer"])
            formatted_qa['metadata'] = {
                'source': 'Specialized QA Generation',
                'topic': qa['topic'],
                'data_type': 'comprehensive_qa',
                'chunk_id': 0
            }
            specialized_pairs.append(formatted_qa)
        
        return specialized_pairs

def generate_qa_dataset(config):
    """Main function to generate QA dataset for Llama 3 fine-tuning."""
    
    # Load processed data
    processed_data_path = os.path.join(config['data_dir'], 'processed_data.json')
    
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data file not found: {processed_data_path}")
        logger.info("Please run process_data.py first to generate processed data")
        return []
    
    with open(processed_data_path, 'r') as f:
        processed_data = json.load(f)
    
    logger.info(f"Loaded {len(processed_data)} processed data items")
    
    # Initialize QA generator
    qa_generator = EVChargingQAGenerator()
    
    # Generate QA pairs from processed data
    qa_pairs = qa_generator.generate_qa_pairs(processed_data)
    
    # Add specialized QA pairs
    specialized_qa = qa_generator.create_specialized_qa_pairs()
    qa_pairs.extend(specialized_qa)
    
    # Shuffle for better training distribution
    random.shuffle(qa_pairs)
    
    # Split into train and eval sets (90/10 split)
    split_idx = int(len(qa_pairs) * 0.9)
    train_data = qa_pairs[:split_idx]
    eval_data = qa_pairs[split_idx:]
    
    logger.info(f"Created {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    
    # Save datasets in multiple formats
    
    # 1. Standard JSON format
    train_path = os.path.join(config['data_dir'], 'train_dataset.json')
    eval_path = os.path.join(config['data_dir'], 'eval_dataset.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    # 2. Hugging Face Dataset format
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    train_dataset.save_to_disk(os.path.join(config['data_dir'], 'train_dataset_hf'))
    eval_dataset.save_to_disk(os.path.join(config['data_dir'], 'eval_dataset_hf'))
    
    # 3. LoRA/QLoRA specific format (instruction format)
    lora_train_data = [{"text": item["text"]} for item in train_data]
    lora_eval_data = [{"text": item["text"]} for item in eval_data]
    
    with open(os.path.join(config['data_dir'], 'train_lora.json'), 'w') as f:
        json.dump(lora_train_data, f, indent=2)
    
    with open(os.path.join(config['data_dir'], 'eval_lora.json'), 'w') as f:
        json.dump(lora_eval_data, f, indent=2)
    
    # 4. JSONL format for streaming
    with open(os.path.join(config['data_dir'], 'train_dataset.jsonl'), 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(os.path.join(config['data_dir'], 'eval_dataset.jsonl'), 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')
    
    # Generate summary statistics
    summary = {
        'total_qa_pairs': len(qa_pairs),
        'train_samples': len(train_data),
        'eval_samples': len(eval_data),
        'topics_covered': {},
        'sources_used': {},
        'avg_question_length': sum(len(item['question']) for item in qa_pairs) / len(qa_pairs),
        'avg_answer_length': sum(len(item['answer']) for item in qa_pairs) / len(qa_pairs)
    }
    
    # Calculate topic and source distributions
    for item in qa_pairs:
        topic = item['metadata']['topic']
        source = item['metadata']['source']
        
        summary['topics_covered'][topic] = summary['topics_covered'].get(topic, 0) + 1
        summary['sources_used'][source] = summary['sources_used'].get(source, 0) + 1
    
    with open(os.path.join(config['data_dir'], 'qa_dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("QA Dataset Generation Complete!")
    logger.info(f"📊 Summary:")
    logger.info(f"  - Total QA pairs: {summary['total_qa_pairs']}")
    logger.info(f"  - Training samples: {summary['train_samples']}")
    logger.info(f"  - Evaluation samples: {summary['eval_samples']}")
    logger.info(f"  - Topics covered: {len(summary['topics_covered'])}")
    logger.info(f"  - Sources used: {len(summary['sources_used'])}")
    logger.info(f"  - Avg question length: {summary['avg_question_length']:.1f} chars")
    logger.info(f"  - Avg answer length: {summary['avg_answer_length']:.1f} chars")
    
    return qa_pairs

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generate_qa_dataset(config) 