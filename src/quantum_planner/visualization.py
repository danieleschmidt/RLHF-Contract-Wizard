"""
Visualization tools for quantum task planning and contract compliance.

Provides interactive visualizations for understanding quantum planning algorithms,
contract compliance status, stakeholder satisfaction, and execution metrics.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
from dataclasses import asdict

from .core import QuantumTaskPlanner, QuantumTask, TaskState
from .contracts import ContractualTaskPlanner, TaskPlanningContext


class QuantumPlannerVisualizer:
    """
    Comprehensive visualization suite for quantum task planning.
    
    Provides static and animated visualizations of quantum states, 
    planning optimization, contract compliance, and execution flow.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_schemes = {
            'quantum_states': {
                TaskState.SUPERPOSITION: '#4B0082',  # Indigo
                TaskState.ENTANGLED: '#FF1493',     # Deep Pink
                TaskState.COLLAPSED: '#32CD32',     # Lime Green
                TaskState.PENDING: '#FFD700',       # Gold
                TaskState.RUNNING: '#FF4500',       # Orange Red
                TaskState.COMPLETED: '#228B22',     # Forest Green
                TaskState.FAILED: '#DC143C',        # Crimson
                TaskState.CANCELLED: '#696969'      # Dim Gray
            },
            'priorities': {
                'high': '#FF4500',    # Orange Red
                'medium': '#FFD700',  # Gold
                'low': '#87CEEB'      # Sky Blue
            },
            'compliance': {
                'compliant': '#228B22',      # Forest Green
                'warning': '#FFD700',        # Gold
                'violation': '#DC143C',      # Crimson
                'unknown': '#696969'         # Dim Gray
            }
        }
    
    def visualize_quantum_state(
        self,
        planner: QuantumTaskPlanner,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize current quantum states of all tasks.
        
        Args:
            planner: Quantum task planner instance
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Quantum Task Planning State Visualization', fontsize=16, fontweight='bold')
        
        tasks = planner.tasks
        if not tasks:
            fig.text(0.5, 0.5, 'No tasks to visualize', ha='center', va='center', fontsize=14)
            return fig
        
        # 1. State Distribution Pie Chart
        state_counts = {}
        for task in tasks.values():
            state_counts[task.state] = state_counts.get(task.state, 0) + 1
        
        if state_counts:
            labels = [state.value.title() for state in state_counts.keys()]
            sizes = list(state_counts.values())
            colors = [self.color_schemes['quantum_states'][state] for state in state_counts.keys()]
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Task State Distribution')
        
        # 2. Quantum Amplitudes and Probabilities
        superposition_tasks = [t for t in tasks.values() if t.state == TaskState.SUPERPOSITION]
        
        if superposition_tasks:
            task_names = [t.name[:15] + '...' if len(t.name) > 15 else t.name for t in superposition_tasks]
            amplitudes = [abs(t.amplitude) for t in superposition_tasks]
            probabilities = [t.probability() for t in superposition_tasks]
            
            x = np.arange(len(task_names))
            width = 0.35
            
            ax2.bar(x - width/2, amplitudes, width, label='Amplitude', alpha=0.7, color='#4B0082')
            ax2.bar(x + width/2, probabilities, width, label='Probability', alpha=0.7, color='#FF1493')
            
            ax2.set_xlabel('Tasks in Superposition')
            ax2.set_ylabel('Values')
            ax2.set_title('Quantum Amplitudes & Probabilities')
            ax2.set_xticks(x)
            ax2.set_xticklabels(task_names, rotation=45, ha='right')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No tasks in superposition', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Quantum Amplitudes & Probabilities')
        
        # 3. Task Priority vs Duration Scatter
        priorities = [t.priority for t in tasks.values()]
        durations = [t.estimated_duration for t in tasks.values()]
        states = [t.state for t in tasks.values()]
        
        for state in set(states):
            state_priorities = [p for p, s in zip(priorities, states) if s == state]
            state_durations = [d for d, s in zip(durations, states) if s == state]
            state_color = self.color_schemes['quantum_states'][state]
            
            ax3.scatter(state_durations, state_priorities, 
                       c=state_color, label=state.value.title(), 
                       alpha=0.7, s=100)
        
        ax3.set_xlabel('Estimated Duration')
        ax3.set_ylabel('Priority')
        ax3.set_title('Task Priority vs Duration by State')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Entanglement Network
        self._plot_entanglement_network(planner, ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_entanglement_network(self, planner: QuantumTaskPlanner, ax: plt.Axes):
        """Plot task entanglement network as a graph."""
        G = nx.Graph()
        
        # Add nodes for all tasks
        for task_id, task in planner.tasks.items():
            G.add_node(task_id, 
                      priority=task.priority,
                      state=task.state,
                      name=task.name)
        
        # Add edges for entanglements
        for (task1, task2), strength in planner.entanglement_matrix.items():
            if abs(strength) > 0.1:  # Only show significant entanglements
                G.add_edge(task1, task2, weight=abs(strength))
        
        if G.nodes():
            # Position nodes using spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes colored by state
            for state in TaskState:
                nodes_with_state = [n for n, d in G.nodes(data=True) if d.get('state') == state]
                if nodes_with_state:
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_state,
                                         node_color=self.color_schemes['quantum_states'][state],
                                         node_size=300, alpha=0.8, ax=ax)
            
            # Draw edges with thickness proportional to entanglement strength
            edges = G.edges(data=True)
            if edges:
                weights = [d.get('weight', 0.1) for u, v, d in edges]
                nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                                     alpha=0.6, edge_color='gray', ax=ax)
            
            # Draw labels
            labels = {n: n[:8] + '...' if len(n) > 8 else n for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Task Entanglement Network')
        ax.axis('off')
    
    def visualize_optimization_history(
        self,
        optimization_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize the optimization process over iterations.
        
        Args:
            optimization_history: History of optimization steps
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Quantum Optimization Process', fontsize=16, fontweight='bold')
        
        if not optimization_history:
            fig.text(0.5, 0.5, 'No optimization history available', ha='center', va='center', fontsize=14)
            return fig
        
        iterations = [step['iteration'] for step in optimization_history]
        fitness_scores = [step['fitness'] for step in optimization_history]
        
        # 1. Fitness Evolution
        ax1.plot(iterations, fitness_scores, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Fitness Evolution During Optimization')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, fitness_scores, 1)
            p = np.poly1d(z)
            ax1.plot(iterations, p(iterations), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax1.legend()
        
        # 2. Convergence Analysis
        if len(fitness_scores) > 1:
            fitness_diff = np.diff(fitness_scores)
            ax2.plot(iterations[1:], np.abs(fitness_diff), 'g-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('|Fitness Change|')
            ax2.set_title('Convergence Rate')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # 3. Solution Quality Distribution
        ax3.hist(fitness_scores, bins=min(20, len(fitness_scores)), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(fitness_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fitness_scores):.3f}')
        ax3.axvline(np.median(fitness_scores), color='orange', linestyle='--',
                   label=f'Median: {np.median(fitness_scores):.3f}')
        ax3.set_xlabel('Fitness Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Solution Quality Distribution')
        ax3.legend()
        
        # 4. Optimization Statistics
        stats_text = f"""
        Total Iterations: {len(optimization_history)}
        Best Fitness: {max(fitness_scores):.4f}
        Final Fitness: {fitness_scores[-1]:.4f}
        Improvement: {fitness_scores[-1] - fitness_scores[0]:.4f}
        Std Deviation: {np.std(fitness_scores):.4f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax4.set_title('Optimization Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_contract_compliance(
        self,
        contractual_planner: ContractualTaskPlanner,
        planning_result: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize contract compliance metrics and stakeholder satisfaction.
        
        Args:
            contractual_planner: Contractual task planner instance
            planning_result: Result from contract-compliant planning
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Contract Compliance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall Compliance Score
        compliance_score = planning_result.get('compliance_score', 0.0)
        contract_fitness = planning_result.get('contract_fitness', 0.0)
        
        scores = [compliance_score, contract_fitness]
        labels = ['Compliance\nScore', 'Contract\nFitness']
        colors = ['#228B22' if s > 0.8 else '#FFD700' if s > 0.6 else '#DC143C' for s in scores]
        
        bars = ax1.bar(labels, scores, color=colors, alpha=0.7)
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Compliance Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Stakeholder Satisfaction
        stakeholder_satisfaction = planning_result.get('stakeholder_satisfaction', 0.5)
        stakeholder_names = list(contractual_planner.stakeholder_weights.keys())
        
        if stakeholder_names:
            # Mock individual stakeholder satisfaction (would be calculated in practice)
            individual_satisfaction = [
                stakeholder_satisfaction * (0.8 + 0.4 * np.random.random()) 
                for _ in stakeholder_names
            ]
            
            colors_satisfaction = [
                '#228B22' if s > 0.8 else '#FFD700' if s > 0.6 else '#DC143C' 
                for s in individual_satisfaction
            ]
            
            bars = ax2.barh(stakeholder_names, individual_satisfaction, color=colors_satisfaction, alpha=0.7)
            ax2.set_xlabel('Satisfaction Score')
            ax2.set_title('Stakeholder Satisfaction')
            ax2.set_xlim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, individual_satisfaction):
                width = bar.get_width()
                ax2.annotate(f'{score:.2f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0),
                           textcoords="offset points",
                           ha='left', va='center')
        else:
            ax2.text(0.5, 0.5, 'No stakeholders defined', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Stakeholder Satisfaction')
        
        # 3. Constraint Violations
        validation_results = planning_result.get('validation_results', {})
        violations = validation_results.get('violations', [])
        
        if violations:
            violation_types = {}
            for violation in violations:
                vtype = violation.get('type', 'unknown')
                severity = violation.get('severity', 'medium')
                
                if vtype not in violation_types:
                    violation_types[vtype] = {'high': 0, 'medium': 0, 'low': 0}
                
                if severity in violation_types[vtype]:
                    violation_types[vtype][severity] += 1
                else:
                    violation_types[vtype]['medium'] += 1
            
            # Stacked bar chart
            types = list(violation_types.keys())
            high_counts = [violation_types[t]['high'] for t in types]
            medium_counts = [violation_types[t]['medium'] for t in types]
            low_counts = [violation_types[t]['low'] for t in types]
            
            ax3.bar(types, high_counts, label='High', color='#DC143C', alpha=0.8)
            ax3.bar(types, medium_counts, bottom=high_counts, label='Medium', color='#FFD700', alpha=0.8)
            ax3.bar(types, low_counts, bottom=np.array(high_counts) + np.array(medium_counts), 
                   label='Low', color='#87CEEB', alpha=0.8)
            
            ax3.set_ylabel('Count')
            ax3.set_title('Constraint Violations by Type')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No violations detected', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, color='green', fontweight='bold')
            ax3.set_title('Constraint Violations by Type')
        
        # 4. Planning Quality Metrics
        quantum_plan = planning_result.get('quantum_plan', {})
        quantum_metrics = quantum_plan.get('quantum_metrics', {})
        
        metrics_text = f"""
        Planning Time: {planning_result.get('planning_time', 0):.3f}s
        Optimization Iterations: {quantum_plan.get('iterations', 0)}
        Converged: {'Yes' if quantum_plan.get('converged', False) else 'No'}
        
        Quantum Metrics:
        - Superposition Tasks: {quantum_metrics.get('superposition_tasks', 0)}
        - Entanglements: {quantum_metrics.get('entanglements', 0)}
        - Avg Probability: {quantum_metrics.get('average_probability', 0):.3f}
        
        Contract Metrics:
        - Stakeholders: {len(stakeholder_names)}
        - Constraints Checked: {planning_result.get('contract_metadata', {}).get('constraints_checked', 0)}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax4.set_title('Planning Quality Metrics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_execution_flow(
        self,
        execution_result: Dict[str, Any],
        tasks: Dict[str, QuantumTask],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize task execution timeline and flow.
        
        Args:
            execution_result: Result from task execution
            tasks: Dictionary of all tasks
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.2))
        fig.suptitle('Task Execution Flow Analysis', fontsize=16, fontweight='bold')
        
        execution_log = execution_result.get('execution_log', [])
        
        if not execution_log:
            fig.text(0.5, 0.5, 'No execution data available', ha='center', va='center', fontsize=14)
            return fig
        
        # 1. Execution Timeline (Gantt Chart)
        task_starts = {}
        task_durations = {}
        task_states = {}
        
        for log_entry in execution_log:
            task_id = log_entry['task_id']
            action = log_entry['action']
            time_point = log_entry['time']
            
            if action == 'started':
                task_starts[task_id] = time_point
            elif action == 'completed' or action == 'failed':
                if task_id in task_starts:
                    task_durations[task_id] = time_point - task_starts[task_id]
                    task_states[task_id] = 'completed' if action == 'completed' else 'failed'
        
        if task_starts:
            # Sort tasks by start time
            sorted_tasks = sorted(task_starts.items(), key=lambda x: x[1])
            task_names = [tid for tid, _ in sorted_tasks]
            
            for i, (task_id, start_time) in enumerate(sorted_tasks):
                duration = task_durations.get(task_id, 0.5)  # Default duration if not found
                state = task_states.get(task_id, 'unknown')
                
                color = '#228B22' if state == 'completed' else '#DC143C' if state == 'failed' else '#696969'
                
                ax1.barh(i, duration, left=start_time, height=0.6, 
                        color=color, alpha=0.7, edgecolor='black')
                
                # Add task label
                ax1.text(start_time + duration/2, i, task_id[:10] + '...' if len(task_id) > 10 else task_id,
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            ax1.set_yticks(range(len(task_names)))
            ax1.set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in task_names])
            ax1.set_xlabel('Time')
            ax1.set_title('Task Execution Timeline (Gantt Chart)')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            completed_patch = patches.Patch(color='#228B22', alpha=0.7, label='Completed')
            failed_patch = patches.Patch(color='#DC143C', alpha=0.7, label='Failed')
            ax1.legend(handles=[completed_patch, failed_patch], loc='upper right')
        
        # 2. Execution Statistics
        completed_tasks = execution_result.get('completed_tasks', [])
        failed_tasks = execution_result.get('failed_tasks', [])
        total_execution_time = execution_result.get('total_execution_time', 0)
        success_rate = execution_result.get('success_rate', 0)
        resource_utilization = execution_result.get('resource_utilization', {})
        
        # Summary pie chart
        if completed_tasks or failed_tasks:
            sizes = [len(completed_tasks), len(failed_tasks)]
            labels = ['Completed', 'Failed']
            colors = ['#228B22', '#DC143C']
            
            # Only show non-zero slices
            non_zero_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
            
            if non_zero_sizes:
                sizes, labels, colors = zip(*non_zero_sizes)
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        
        # Add execution summary text
        summary_text = f"""
        Execution Summary:
        - Total Time: {total_execution_time:.2f}s
        - Success Rate: {success_rate:.1%}
        - Tasks Completed: {len(completed_tasks)}
        - Tasks Failed: {len(failed_tasks)}
        
        Resource Utilization:
        """
        
        for resource, utilization in resource_utilization.items():
            summary_text += f"- {resource}: {utilization:.1%}\n"
        
        ax2.text(1.3, 0.5, summary_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        ax2.set_title('Execution Results Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_comprehensive_dashboard(
        self,
        contractual_planner: ContractualTaskPlanner,
        planning_result: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard combining all visualizations.
        
        Args:
            contractual_planner: Contractual task planner instance
            planning_result: Result from contract-compliant planning
            execution_result: Optional execution results
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Quantum Task Planning & Contract Compliance Dashboard', fontsize=20, fontweight='bold')
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Quantum state visualization
        ax_quantum = fig.add_subplot(gs[0, :2])
        self._plot_quantum_state_summary(contractual_planner.quantum_planner, ax_quantum)
        
        # Contract compliance
        ax_compliance = fig.add_subplot(gs[0, 2:])
        self._plot_compliance_summary(contractual_planner, planning_result, ax_compliance)
        
        # Optimization history
        quantum_plan = planning_result.get('quantum_plan', {})
        optimization_history = quantum_plan.get('optimization_history', [])
        
        ax_optimization = fig.add_subplot(gs[1, :2])
        if optimization_history:
            iterations = [step['iteration'] for step in optimization_history]
            fitness_scores = [step['fitness'] for step in optimization_history]
            ax_optimization.plot(iterations, fitness_scores, 'b-', linewidth=2, marker='o', markersize=3)
            ax_optimization.set_xlabel('Iteration')
            ax_optimization.set_ylabel('Fitness Score')
            ax_optimization.set_title('Optimization Progress')
            ax_optimization.grid(True, alpha=0.3)
        else:
            ax_optimization.text(0.5, 0.5, 'No optimization history', ha='center', va='center', transform=ax_optimization.transAxes)
            ax_optimization.set_title('Optimization Progress')
        
        # Execution timeline
        ax_execution = fig.add_subplot(gs[1, 2:])
        if execution_result:
            self._plot_execution_summary(execution_result, ax_execution)
        else:
            ax_execution.text(0.5, 0.5, 'No execution data', ha='center', va='center', transform=ax_execution.transAxes)
            ax_execution.set_title('Execution Summary')
        
        # Metrics summary
        ax_metrics = fig.add_subplot(gs[2, :])
        self._plot_metrics_summary(contractual_planner, planning_result, execution_result, ax_metrics)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_quantum_state_summary(self, planner: QuantumTaskPlanner, ax: plt.Axes):
        """Plot summary of quantum states."""
        state_summary = planner.get_quantum_state_summary()
        
        state_dist = state_summary.get('state_distribution', {})
        if state_dist:
            states = list(state_dist.keys())
            counts = list(state_dist.values())
            colors = [self.color_schemes['quantum_states'].get(
                TaskState(state), '#696969') for state in states]
            
            bars = ax.bar(states, counts, color=colors, alpha=0.7)
            ax.set_ylabel('Task Count')
            ax.set_title('Quantum State Distribution')
            ax.tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No quantum state data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Quantum State Distribution')
    
    def _plot_compliance_summary(
        self, 
        contractual_planner: ContractualTaskPlanner,
        planning_result: Dict[str, Any],
        ax: plt.Axes
    ):
        """Plot contract compliance summary."""
        compliance_score = planning_result.get('compliance_score', 0.0)
        stakeholder_satisfaction = planning_result.get('stakeholder_satisfaction', 0.0)
        
        categories = ['Compliance\nScore', 'Stakeholder\nSatisfaction']
        scores = [compliance_score, stakeholder_satisfaction]
        colors = ['#228B22' if s > 0.8 else '#FFD700' if s > 0.6 else '#DC143C' for s in scores]
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Contract Compliance Summary')
        ax.set_ylim(0, 1)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    def _plot_execution_summary(self, execution_result: Dict[str, Any], ax: plt.Axes):
        """Plot execution results summary."""
        completed = len(execution_result.get('completed_tasks', []))
        failed = len(execution_result.get('failed_tasks', []))
        
        if completed > 0 or failed > 0:
            labels = []
            sizes = []
            colors = []
            
            if completed > 0:
                labels.append(f'Completed ({completed})')
                sizes.append(completed)
                colors.append('#228B22')
            
            if failed > 0:
                labels.append(f'Failed ({failed})')
                sizes.append(failed)
                colors.append('#DC143C')
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, 'No execution results', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Execution Results')
    
    def _plot_metrics_summary(
        self,
        contractual_planner: ContractualTaskPlanner,
        planning_result: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]],
        ax: plt.Axes
    ):
        """Plot comprehensive metrics summary."""
        # Prepare metrics text
        quantum_plan = planning_result.get('quantum_plan', {})
        contract_metadata = planning_result.get('contract_metadata', {})
        
        metrics_text = f"""
PLANNING METRICS:
• Planning Time: {planning_result.get('planning_time', 0):.3f}s
• Optimization Iterations: {quantum_plan.get('iterations', 0)}
• Converged: {'✓' if quantum_plan.get('converged', False) else '✗'}
• Fitness Score: {quantum_plan.get('fitness_score', 0):.3f}

CONTRACT METRICS:
• Contract: {contract_metadata.get('contract_name', 'N/A')} v{contract_metadata.get('contract_version', 'N/A')}
• Stakeholders: {len(contract_metadata.get('stakeholders', []))}
• Constraints Checked: {contract_metadata.get('constraints_checked', 0)}
• Compliance Score: {planning_result.get('compliance_score', 0):.3f}

QUANTUM METRICS:
• Total Tasks: {len(contractual_planner.quantum_planner.tasks)}
• Superposition Tasks: {quantum_plan.get('quantum_metrics', {}).get('superposition_tasks', 0)}
• Entanglements: {quantum_plan.get('quantum_metrics', {}).get('entanglements', 0)}
• Avg Probability: {quantum_plan.get('quantum_metrics', {}).get('average_probability', 0):.3f}
        """
        
        if execution_result:
            metrics_text += f"""
EXECUTION METRICS:
• Total Execution Time: {execution_result.get('total_execution_time', 0):.3f}s
• Success Rate: {execution_result.get('success_rate', 0):.1%}
• Completed Tasks: {len(execution_result.get('completed_tasks', []))}
• Failed Tasks: {len(execution_result.get('failed_tasks', []))}
            """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Comprehensive Metrics Summary')
        ax.axis('off')
    
    def export_dashboard_data(
        self,
        contractual_planner: ContractualTaskPlanner,
        planning_result: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None,
        export_path: str = "quantum_planning_dashboard.json"
    ):
        """Export dashboard data to JSON for external analysis or web dashboards."""
        
        # Prepare exportable data
        dashboard_data = {
            'timestamp': time.time(),
            'contract_info': {
                'name': contractual_planner.contract.metadata.name,
                'version': contractual_planner.contract.metadata.version,
                'stakeholders': {
                    name: {
                        'weight': stakeholder.weight,
                        'voting_power': stakeholder.voting_power,
                        'address': stakeholder.address
                    }
                    for name, stakeholder in contractual_planner.contract.stakeholders.items()
                },
                'constraints': {
                    name: {
                        'description': constraint.description,
                        'severity': constraint.severity,
                        'enabled': constraint.enabled
                    }
                    for name, constraint in contractual_planner.contract.constraints.items()
                }
            },
            'planning_results': planning_result,
            'quantum_state_summary': contractual_planner.quantum_planner.get_quantum_state_summary(),
            'tasks': {
                task_id: {
                    'name': task.name,
                    'priority': task.priority,
                    'estimated_duration': task.estimated_duration,
                    'state': task.state.value,
                    'probability': task.probability(),
                    'dependencies': list(task.dependencies),
                    'entangled_tasks': list(task.entangled_tasks)
                }
                for task_id, task in contractual_planner.quantum_planner.tasks.items()
            }
        }
        
        if execution_result:
            dashboard_data['execution_results'] = execution_result
        
        # Write to JSON file
        with open(export_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return export_path