import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import os
import sys

# Add the workspace root to the path to ensure imports work correctly
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_root)

from RAG.chatBot.rag_models import get_rag_model
from RAG.Ret_Gen_Evaluations.retrieval_evaluation import evaluate_retrieval
from RAG.Ret_Gen_Evaluations.generation_evaluation import evaluate_generation

def main():
    # Initialize your RAG model
    print("Initializing RAG model...")
    
    rag_model = get_rag_model()
    
    # Load test dataset
    print("Loading test dataset...")
    test_db_path = Path(__file__).parent.parent.parent / "TestDatasets/TEST_DATABASE_2.json"
    with open(test_db_path, 'r') as f:
        test_data = json.load(f)
    
    # Extract test questions and ground truth
    test_questions = [item['question'] for item in test_data]
    reference_answers = {item['question']: item['expected_answer'] for item in test_data}
    ground_truth_docs = {item['question']: item['relevant_chunk_ids'] for item in test_data}
    
    # Create results directory
    results_dir = Path(__file__).parent / f"EVALUATION_OPEN_MINDED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate retrieval with multiple configurations
    print("\n=== Evaluating Retrieval Component ===")
    print("Testing multiple k and threshold configurations...")
    try:
        # Run retrieval evaluation with multiple configurations
        # Note: None threshold means no threshold (original behavior - accepts all documents)
        retrieval_results = evaluate_retrieval(
            rag_model, 
            test_questions, 
            ground_truth_docs,
            k_values=[3, 5, 7, 10],
            threshold_values=[None, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        
        # Save retrieval results
        with open(results_dir / "retrieval_results.json", 'w') as f:
            json.dump(retrieval_results, f, indent=2)
        
        # Print comparison table
        print("\n" + "="*80)
        print("RETRIEVAL PERFORMANCE COMPARISON ACROSS CONFIGURATIONS")
        print("="*80)
        print(f"{'Configuration':<25} {'Precision':<12} {'Recall':<12} {'MRR':<12} {'NDCG':<12} {'Hit Rate':<12}")
        print("-"*80)
        
        # Sort configurations for better readability
        sorted_configs = sorted(retrieval_results['configurations'].items())
        for config_name, metrics in sorted_configs:
            print(f"{config_name:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['mrr']:<12.4f} {metrics['ndcg']:<12.4f} {metrics['hit_rate']:<12.4f}")
        
        # Find and print best configurations for each metric
        print("\n" + "="*80)
        print("BEST PERFORMING CONFIGURATIONS PER METRIC")
        print("="*80)
        
        configs = retrieval_results['configurations']
        for metric in ['precision', 'recall', 'mrr', 'ndcg', 'hit_rate']:
            best_config = max(configs.items(), key=lambda x: x[1][metric])
            print(f"{metric.upper():<15}: {best_config[0]} (score: {best_config[1][metric]:.4f})")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"Error in retrieval evaluation: {e}")
        import traceback
        traceback.print_exc()
        retrieval_results = {
            "configurations": {},
            "metadata": {},
            "error": str(e)
        }
    
    # Ask user for evaluation mode
    print("\nChoose generation evaluation mode:")
    print("1. Use ground truth documents as fallback when retrieval fails")
    print("2. Only use actually retrieved documents (skip failed retrievals)")
    while True:
        mode = input("Enter mode (1 or 2): ").strip()
        if mode in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    use_ground_truth_fallback = (mode == '1')
    
    # Evaluate generation for each model
    model_names = ["gpt4o-mini", "gpt3.5-turbo", "llama-70b", "llama-vision"]
    generation_results = {}
    
    # Ask user how many questions to test for generation
    num_questions = 5  # Default
    try:
        user_input = input(f"\nHow many questions to test for generation? (default: {num_questions}, max: {len(test_questions)}): ")
        if user_input.strip():
            num_questions = min(int(user_input), len(test_questions))
    except ValueError:
        print(f"Using default: {num_questions} questions")
    
    for model_name in model_names:
        print(f"\n=== Evaluating Generation Component ({model_name}) ===")
        print(f"Testing with {num_questions} questions...")
        
        generation_results[model_name] = evaluate_generation(
            rag_model, 
            test_questions[:num_questions], 
            reference_answers, 
            model_name,
            ground_truth_docs,
            use_ground_truth_fallback=use_ground_truth_fallback
        )
        
        # Check for evaluation errors
        if generation_results[model_name].get('error'):
            print(f"Error in {model_name} evaluation: {generation_results[model_name]['error']}")
            continue
        
        # Print statistics about processed questions
        stats = generation_results[model_name].get('processing_stats', {})
        if stats:
            print(f"\nProcessing Statistics:")
            print(f"- Total questions attempted: {stats.get('total_questions', 0)}")
            print(f"- Successfully processed: {stats.get('processed_questions', 0)}")
            print(f"- Skipped (no retrieval): {stats.get('skipped_questions', 0)}")
        
        # Save generation results
        with open(results_dir / f"generation_results_{model_name}.json", 'w') as f:
            json.dump(generation_results[model_name], f, indent=2)
        
        print(f"\nGeneration Results ({model_name}):")
        print(json.dumps({k: v for k, v in generation_results[model_name].items() if k != 'processing_stats'}, indent=2))
    
    # Save combined results
    combined_results = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_test_questions": len(test_questions),
            "num_generation_questions": num_questions,
            "evaluation_mode": "ground_truth_fallback" if use_ground_truth_fallback else "retrieval_only"
        }
    }
    
    with open(results_dir / "combined_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nAll evaluation results saved to {results_dir}")
    
    # Generate summary report
    generate_summary_report(combined_results, results_dir)

def generate_summary_report(results, results_dir):
    """Generate a human-readable summary report of the evaluation results"""
    report = []
    report.append("# RAG System Evaluation Summary")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Retrieval metrics - now with multiple configurations
    report.append("## Retrieval Performance")
    retrieval = results["retrieval"]
    
    # Check if we have the new multi-configuration format
    if 'configurations' in retrieval:
        report.append(f"Tested {retrieval['metadata']['total_configurations']} configurations:")
        report.append(f"- K values: {retrieval['metadata']['k_values']}")
        report.append(f"- Threshold values: {retrieval['metadata']['threshold_values']}")
        report.append("")
        
        # Create table header
        report.append("### Configuration Results")
        report.append("")
        report.append("| Configuration | Precision | Recall | MRR | NDCG | Hit Rate |")
        report.append("|---------------|-----------|--------|-----|------|----------|")
        
        # Add each configuration
        for config_name, metrics in sorted(retrieval['configurations'].items()):
            report.append(f"| {config_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                         f"{metrics['mrr']:.4f} | {metrics['ndcg']:.4f} | {metrics['hit_rate']:.4f} |")
        
        report.append("")
        report.append("### Best Performing Configurations")
        report.append("")
        
        # Find best for each metric
        configs = retrieval['configurations']
        for metric in ['precision', 'recall', 'mrr', 'ndcg', 'hit_rate']:
            best_config = max(configs.items(), key=lambda x: x[1][metric])
            report.append(f"- **{metric.upper()}**: {best_config[0]} (score: {best_config[1][metric]:.4f})")
    else:
        # Old format for backwards compatibility
        report.append(f"- Precision: {retrieval.get('precision', 0):.4f}")
        report.append(f"- Recall: {retrieval.get('recall', 0):.4f}")
        report.append(f"- MRR (Mean Reciprocal Rank): {retrieval.get('mrr', 0):.4f}")
        report.append(f"- Hit Rate: {retrieval.get('hit_rate', 0):.4f}")
        report.append(f"- NDCG: {retrieval.get('ndcg', 0):.4f}")
    
    report.append("")
    
    # Generation metrics
    report.append("## Generation Performance")
    for model_name, gen_results in results["generation"].items():
        report.append(f"### {model_name.upper()}")
        
        # Check for errors
        if gen_results.get('error'):
            report.append(f"Error: {gen_results['error']}")
            report.append("")
            continue
        
        report.append("#### Semantic Metrics")
        report.append(f"- BERTScore Precision: {gen_results['bert_precision']:.4f}")
        report.append(f"- BERTScore Recall: {gen_results['bert_recall']:.4f}")
        report.append(f"- BERTScore F1: {gen_results['bert_f1']:.4f}")
        report.append(f"- Semantic Similarity: {gen_results['semantic_similarity']:.4f}")
        report.append("")
        
        report.append("#### ROUGE Metrics")
        report.append(f"- ROUGE-1 F1: {gen_results['rouge1']:.4f}")
        report.append(f"- ROUGE-2 F1: {gen_results['rouge2']:.4f}")
        report.append(f"- ROUGE-L F1: {gen_results['rougeL']:.4f}")
        report.append("")
        
        report.append("#### LLM-as-judge Metrics (custom implementation)")
        report.append(f"- Answer Relevance: {gen_results['answer_relevance']:.4f}")
        report.append(f"- Factual Accuracy: {gen_results['factual_accuracy']:.4f}")
        report.append(f"- Groundedness: {gen_results['groundedness']:.4f}")
        
        # Add GEval scores if available
        if 'geval_score' in gen_results:
            report.append("")
            report.append("#### GEval Metrics (LLM-as-judge using deepeval)")
            report.append(f"- GEval Correctness: {gen_results['geval_score']:.4f}")
            
            # Add the comparison metrics
            if 'geval_relevance' in gen_results:
                report.append(f"- GEval Relevance: {gen_results['geval_relevance']:.4f}")
            if 'geval_accuracy' in gen_results:
                report.append(f"- GEval Accuracy: {gen_results['geval_accuracy']:.4f}")
            if 'geval_groundedness' in gen_results:
                report.append(f"- GEval Groundedness: {gen_results['geval_groundedness']:.4f}")
                
            # Add comparison summary
            report.append("")
            report.append("#### Comparison between LLM evaluation methods")
            if 'geval_relevance' in gen_results and 'answer_relevance' in gen_results:
                diff = gen_results['geval_relevance'] - gen_results['answer_relevance']
                report.append(f"- Relevance difference: {diff:.4f} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")
            if 'geval_accuracy' in gen_results and 'factual_accuracy' in gen_results:
                diff = gen_results['geval_accuracy'] - gen_results['factual_accuracy']
                report.append(f"- Accuracy difference: {diff:.4f} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")
            if 'geval_groundedness' in gen_results and 'groundedness' in gen_results:
                diff = gen_results['geval_groundedness'] - gen_results['groundedness']
                report.append(f"- Groundedness difference: {diff:.4f} ({'+' if diff >= 0 else ''}{diff*100:.1f}%)")
            
        report.append("")
    
    # Write report to file
    with open(results_dir / "evaluation_summary.md", 'w') as f:
        f.write("\n".join(report))
    
    print(f"Summary report generated: {results_dir / 'evaluation_summary.md'}")

if __name__ == "__main__":
    main()