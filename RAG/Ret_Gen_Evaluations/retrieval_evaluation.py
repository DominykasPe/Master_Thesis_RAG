import numpy as np

def evaluate_retrieval(rag_model, test_questions, ground_truth_docs, 
                      k_values=[3, 5, 7, 10], threshold_values=[None, 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Evaluate retrieval performance across multiple configurations.
    
    Args:
        rag_model: The RAG model instance
        test_questions: List of test questions
        ground_truth_docs: Dictionary mapping questions to relevant document IDs
        k_values: List of k values to test (number of documents to retrieve)
        threshold_values: List of score threshold values to test (None means no threshold)
    
    Returns:
        Dictionary with results for all configurations and metadata
    """
    
    # Store original settings to restore later
    original_k = rag_model.retriever.search_kwargs["k"]
    original_threshold = rag_model.retriever.search_kwargs.get("score_threshold", 0.7)
    
    # Store all configuration results
    all_configurations = {}
    
    # Test each combination of k and threshold
    for k in k_values:
        for threshold in threshold_values:
            # Create config name - handle None threshold specially
            config_name = f"k={k}_threshold={'None' if threshold is None else threshold}"
            print(f"\n{'='*60}")
            print(f"Testing configuration: {config_name}")
            print(f"{'='*60}")
            
            # Set configuration
            rag_model.retriever.search_kwargs["k"] = k
            if threshold is None:
                # Remove threshold completely (original behavior)
                rag_model.retriever.search_kwargs.pop("score_threshold", None)
            else:
                rag_model.retriever.search_kwargs["score_threshold"] = threshold
            
            # Initialize results for this configuration
            results = {
                'precision': [],
                'recall': [],
                'mrr': [],
                'hit_rate': [],
                'ndcg': []
            }
            
            # Add counters for total documents
            total_retrieved_docs = 0
            
            for question in test_questions:
                print(f"\nEvaluating retrieval for question: {question}")
                
                retrieved_docs = rag_model.retriever.get_relevant_documents(question)
                total_retrieved_docs += len(retrieved_docs)  # Add to counter
                retrieved_ids = []
                for doc in retrieved_docs:
                    if doc.metadata.get('source_type') == 'lessons_learned':
                        retrieved_ids.append(doc.metadata.get('url', ''))
                    else:
                        retrieved_ids.append(doc.metadata.get('chunk_id', ''))
                
                # Get ground truth for this question
                relevant_ids = ground_truth_docs.get(question, [])
                
                # DEBUG: Print expected vs. retrieved IDs
                print(f"  Expected document IDs: {relevant_ids}")
                print(f"  Retrieved document IDs: {retrieved_ids}")
                
                # Enhanced partial matching
                matched_ids = set()
                for rel_id in relevant_ids:
                    for ret_id in retrieved_ids:
                        # Check for exact match first
                        if rel_id == ret_id:
                            matched_ids.add(rel_id)
                        # Then check for partial match
                        elif rel_id in ret_id or ret_id in rel_id:
                            print(f"  Partial match found: '{rel_id}' ~ '{ret_id}'")
                            matched_ids.add(rel_id)
                
                # Calculate metrics using matched IDs
                precision = len(matched_ids) / len(retrieved_ids) if retrieved_ids else 0
                recall = len(matched_ids) / len(relevant_ids) if relevant_ids else 0
                
                # MRR calculation with partial matching
                mrr = 0
                for i, ret_id in enumerate(retrieved_ids):
                    for rel_id in relevant_ids:
                        if rel_id == ret_id or rel_id in ret_id or ret_id in rel_id:
                            mrr = 1.0 / (i + 1)
                            break
                    if mrr > 0:
                        break
                
                # Hit rate based on matched IDs
                hit_rate = 1 if matched_ids else 0
                
                # NDCG calculation
                ndcg = calculate_ndcg(retrieved_ids, relevant_ids)
                
                # Store results
                results['precision'].append(precision)
                results['recall'].append(recall)
                results['mrr'].append(mrr)
                results['hit_rate'].append(hit_rate)
                results['ndcg'].append(ndcg)
                
                print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, MRR: {mrr:.2f}, NDCG: {ndcg:.2f}")
                
                # Print document details for debugging
                print("  Retrieved documents details:")
                for i, doc in enumerate(retrieved_docs):
                    source_type = doc.metadata.get('source_type', 'unknown')
                    doc_id = doc.metadata.get('chunk_id', doc.metadata.get('url', 'unknown'))
                    title = doc.metadata.get('title', 'unknown')
                    print(f"    {i+1}. Type: {source_type}, ID: {doc_id}, Title: {title}")
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"       Content: {content_preview}")
            
            # Calculate and print statistics for this configuration
            total_relevant_docs = sum(len(ground_truth_docs.get(q, [])) for q in test_questions)
            avg_docs_per_query = total_retrieved_docs / len(test_questions)
            print("\n=== Retrieval Statistics ===")
            print(f"Total documents retrieved: {total_retrieved_docs}")
            print(f"Total relevant documents: {total_relevant_docs}")
            print(f"Average documents per query: {avg_docs_per_query:.2f}")
            print(f"Average relevant documents per query: {total_relevant_docs/len(test_questions):.2f}")
            
            # Calculate final results for this configuration
            config_results = {}
            for metric in results:
                config_results[metric] = sum(results[metric]) / len(results[metric]) if results[metric] else 0.0
            
            # Add configuration metadata
            config_results['total_retrieved_docs'] = total_retrieved_docs
            config_results['avg_docs_per_query'] = avg_docs_per_query
            config_results['questions_with_hits'] = sum(results['hit_rate'])
            
            # Store in all_configurations
            all_configurations[config_name] = config_results
            
            # Print summary for this configuration
            print("\n=== Configuration Summary ===")
            print(f"Configuration: {config_name}")
            print(f"Total questions evaluated: {len(test_questions)}")
            print(f"Questions with at least one relevant document retrieved: {sum(results['hit_rate'])}")
            print(f"Average precision: {config_results['precision']:.4f}")
            print(f"Average recall: {config_results['recall']:.4f}")
            print(f"Average MRR: {config_results['mrr']:.4f}")
            print(f"Average NDCG: {config_results['ndcg']:.4f}")
    
    # Restore original settings
    rag_model.retriever.search_kwargs["k"] = original_k
    if original_threshold is not None:
        rag_model.retriever.search_kwargs["score_threshold"] = original_threshold
    
    # Return all results with metadata
    return {
        'configurations': all_configurations,
        'metadata': {
            'k_values': k_values,
            'threshold_values': threshold_values,
            'total_configurations': len(all_configurations),
            'num_test_questions': len(test_questions)
        }
    }

def calculate_ndcg(retrieved_ids, relevant_ids, k=None):
    """Calculate NDCG for a single query."""
    if k is None:
        k = len(retrieved_ids)
    
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    dcg = 0
    idcg = 0
    
    # Calculate DCG
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1 if doc_id in relevant_ids else 0
        dcg += rel / np.log2(i + 2)
    
    # Calculate IDCG
    for i in range(min(len(relevant_ids), k)):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0
