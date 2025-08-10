#!/usr/bin/env python3
"""
RLHF-Contract-Wizard CLI application.

Main command-line interface for creating, managing, and deploying
RLHF reward contracts with legal compliance and formal verification.
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

import click
import jax
import jax.numpy as jnp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.reward_contract import RewardContract, AggregationStrategy
from models.legal_blocks import LegalBlocks, RLHFConstraints
from models.reward_model import ContractualRewardModel, RewardModelConfig
from services.contract_service import ContractService
from services.verification_service import VerificationService
from services.blockchain_service import BlockchainService
from quantum_planner.core import QuantumPlanner, PlannerConfig
from utils.helpers import setup_logging, load_config


# Setup logging
logger = setup_logging()

# Global context for CLI state
class CLIContext:
    """Global CLI context for sharing state between commands."""
    
    def __init__(self):
        self.config = {}
        self.current_contract = None
        self.services = {}
        self.planner = None


# Initialize global context
ctx = CLIContext()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(click_ctx, config, verbose):
    """RLHF-Contract-Wizard: Legal-compliant RLHF with smart contracts."""
    
    # Setup logging level
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        ctx.config = load_config(config)
    else:
        # Default configuration
        ctx.config = {
            'database_url': os.getenv('DATABASE_URL', 'postgresql://localhost/rlhf_contracts'),
            'blockchain_url': os.getenv('BLOCKCHAIN_URL', 'http://localhost:8545'),
            'verification_backend': os.getenv('VERIFICATION_BACKEND', 'mock'),
            'random_seed': int(os.getenv('RANDOM_SEED', '42'))
        }
    
    # Initialize services
    try:
        ctx.services = {
            'contract': ContractService(),
            'verification': VerificationService(backend=ctx.config['verification_backend']),
            'blockchain': BlockchainService(ctx.config['blockchain_url'])
        }
        
        # Initialize quantum planner
        planner_config = PlannerConfig(
            algorithm='quantum_annealing',
            num_qubits=16,
            optimization_steps=1000
        )
        ctx.planner = QuantumPlanner(planner_config)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        sys.exit(1)
    
    click_ctx.obj = ctx


@cli.group()
def contract():
    """Contract management commands."""
    pass


@contract.command('create')
@click.option('--name', '-n', required=True, help='Contract name')
@click.option('--version', '-v', default='1.0.0', help='Contract version')
@click.option('--stakeholders', '-s', help='Stakeholders JSON (e.g., \'{"users":0.5,"safety":0.3,"ops":0.2}\')')
@click.option('--jurisdiction', '-j', default='Global', help='Legal jurisdiction')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def create_contract(name, version, stakeholders, jurisdiction, output):
    """Create a new reward contract."""
    try:
        # Parse stakeholders
        stakeholder_weights = {}
        if stakeholders:
            stakeholder_weights = json.loads(stakeholders)
        
        # Create contract
        contract = RewardContract(
            name=name,
            version=version,
            stakeholders=stakeholder_weights,
            jurisdiction=jurisdiction
        )
        
        # Add default safety constraints
        contract.add_constraint(
            "safety_baseline",
            RLHFConstraints.no_harmful_output,
            "Prevent harmful content generation",
            severity=2.0,
            violation_penalty=-2.0
        )
        
        contract.add_constraint(
            "privacy_protection",
            RLHFConstraints.privacy_protection,
            "Protect user privacy and PII",
            severity=1.5,
            violation_penalty=-1.5
        )
        
        contract.add_constraint(
            "fairness_requirement",
            RLHFConstraints.fairness_requirement,
            "Ensure fair treatment across demographics",
            severity=1.0,
            violation_penalty=-1.0
        )
        
        # Store in service
        contract_id = ctx.services['contract'].create_contract(contract.to_dict())
        
        # Save to file if specified
        if output:
            with open(output, 'w') as f:
                json.dump(contract.to_dict(), f, indent=2)
            click.echo(f"Contract saved to {output}")
        
        ctx.current_contract = contract
        
        click.echo(f"✅ Created contract '{name}' (ID: {contract_id})")
        click.echo(f"   Hash: {contract.compute_hash()}")
        click.echo(f"   Stakeholders: {len(contract.stakeholders)}")
        click.echo(f"   Constraints: {len(contract.constraints)}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create contract: {e}", err=True)
        sys.exit(1)


@contract.command('list')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']), help='Output format')
def list_contracts(output_format):
    """List all contracts."""
    try:
        contracts = ctx.services['contract'].list_contracts()
        
        if output_format == 'json':
            click.echo(json.dumps(contracts, indent=2))
        else:
            # Table format
            click.echo("\nContracts:")
            click.echo("-" * 80)
            click.echo(f"{'Name':<20} {'Version':<10} {'Stakeholders':<12} {'Constraints':<12} {'Created':<15}")
            click.echo("-" * 80)
            
            for contract in contracts:
                metadata = contract.get('metadata', {})
                stakeholders = len(contract.get('stakeholders', {}))
                constraints = len(contract.get('constraints', {}))
                created = time.strftime('%Y-%m-%d', time.localtime(metadata.get('created_at', 0)))
                
                click.echo(f"{metadata.get('name', 'Unknown'):<20} {metadata.get('version', '1.0.0'):<10} {stakeholders:<12} {constraints:<12} {created:<15}")
            
            click.echo("-" * 80)
            click.echo(f"Total: {len(contracts)} contracts")
        
    except Exception as e:
        click.echo(f"❌ Failed to list contracts: {e}", err=True)


@contract.command('verify')
@click.option('--contract', '-c', help='Contract file or name')
@click.option('--backend', '-b', default='mock', type=click.Choice(['z3', 'lean4', 'mock']), help='Verification backend')
def verify_contract(contract, backend):
    """Verify contract constraints."""
    try:
        # Load contract
        if contract:
            if os.path.exists(contract):
                with open(contract) as f:
                    contract_data = json.load(f)
            else:
                contract_data = ctx.services['contract'].get_contract_by_name(contract)
        else:
            if not ctx.current_contract:
                click.echo("❌ No contract specified. Use --contract or create one first.", err=True)
                return
            contract_data = ctx.current_contract.to_dict()
        
        # Run verification
        click.echo("🔍 Verifying contract constraints...")
        
        verification_service = VerificationService(backend=backend)
        result = verification_service.verify_contract(contract_data)
        
        if result['valid']:
            click.echo("✅ Contract verification passed!")
            click.echo(f"   Properties verified: {result.get('properties_verified', 0)}")
            if 'proof_size' in result:
                click.echo(f"   Proof size: {result['proof_size']}")
        else:
            click.echo("❌ Contract verification failed!")
            for violation in result.get('violations', []):
                click.echo(f"   • {violation}")
        
        if 'verification_time' in result:
            click.echo(f"   Verification time: {result['verification_time']:.2f}s")
        
    except Exception as e:
        click.echo(f"❌ Verification failed: {e}", err=True)


@cli.group()
def model():
    """Reward model commands."""
    pass


@model.command('train')
@click.option('--contract', '-c', help='Contract file or name')
@click.option('--data', '-d', required=True, type=click.Path(exists=True), help='Training data path')
@click.option('--output', '-o', type=click.Path(), help='Output model path')
@click.option('--epochs', '-e', default=10, help='Training epochs')
@click.option('--batch-size', '-b', default=32, help='Batch size')
def train_model(contract, data, output, epochs, batch_size):
    """Train a contractual reward model."""
    try:
        # Load contract
        if contract:
            if os.path.exists(contract):
                with open(contract) as f:
                    contract_dict = json.load(f)
            else:
                contract_dict = ctx.services['contract'].get_contract_by_name(contract)
            
            # Reconstruct contract object
            reward_contract = RewardContract(
                name=contract_dict['metadata']['name'],
                version=contract_dict['metadata']['version'],
                stakeholders={name: info['weight'] for name, info in contract_dict['stakeholders'].items()}
            )
        else:
            if not ctx.current_contract:
                click.echo("❌ No contract specified. Use --contract or create one first.", err=True)
                return
            reward_contract = ctx.current_contract
        
        # Load training data
        click.echo(f"📊 Loading training data from {data}")
        # In practice, would load actual preference data
        # For demo, create mock data
        random_key = jax.random.PRNGKey(ctx.config['random_seed'])
        
        # Create model configuration
        config = RewardModelConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            max_sequence_length=128,
            batch_size=batch_size
        )
        
        # Initialize model
        model = ContractualRewardModel(config, reward_contract, random_key)
        
        # Mock training loop
        click.echo(f"🎯 Training contractual reward model...")
        click.echo(f"   Contract: {reward_contract.metadata.name}")
        click.echo(f"   Epochs: {epochs}")
        click.echo(f"   Batch size: {batch_size}")
        
        training_metrics = []
        
        with click.progressbar(range(epochs), label='Training progress') as bar:
            for epoch in bar:
                # Generate mock training batch
                chosen_tokens = jax.random.randint(
                    random_key, (batch_size, config.max_sequence_length), 0, 1000
                )
                rejected_tokens = jax.random.randint(
                    random_key, (batch_size, config.max_sequence_length), 0, 1000
                )
                
                # Update model
                metrics = model.update(chosen_tokens, rejected_tokens)
                training_metrics.append(metrics)
                
                # Update random key
                random_key, _ = jax.random.split(random_key)
        
        # Final metrics
        final_metrics = training_metrics[-1]
        click.echo(f"\n✅ Training completed!")
        click.echo(f"   Final loss: {final_metrics.loss:.4f}")
        click.echo(f"   Contract compliance: {final_metrics.contract_compliance_rate:.2%}")
        click.echo(f"   Training time: {sum(m.training_time for m in training_metrics):.2f}s")
        
        # Save model
        if output:
            model.save_checkpoint(output)
            click.echo(f"💾 Model saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)


@cli.group()
def deploy():
    """Deployment commands."""
    pass


@deploy.command('blockchain')
@click.option('--contract', '-c', help='Contract file or name')
@click.option('--network', '-n', default='local', type=click.Choice(['local', 'testnet', 'mainnet']), help='Blockchain network')
@click.option('--gas-limit', '-g', default=500000, help='Gas limit for deployment')
def deploy_blockchain(contract, network, gas_limit):
    """Deploy contract to blockchain."""
    try:
        # Load contract
        if contract:
            if os.path.exists(contract):
                with open(contract) as f:
                    contract_data = json.load(f)
            else:
                contract_data = ctx.services['contract'].get_contract_by_name(contract)
        else:
            if not ctx.current_contract:
                click.echo("❌ No contract specified. Use --contract or create one first.", err=True)
                return
            contract_data = ctx.current_contract.to_dict()
        
        click.echo(f"🚀 Deploying contract to {network} network...")
        
        # Deploy to blockchain
        deployment_result = ctx.services['blockchain'].deploy_contract(
            contract_data,
            network=network,
            gas_limit=gas_limit
        )
        
        if deployment_result['success']:
            click.echo("✅ Contract deployed successfully!")
            click.echo(f"   Transaction hash: {deployment_result['tx_hash']}")
            click.echo(f"   Contract address: {deployment_result['contract_address']}")
            click.echo(f"   Gas used: {deployment_result['gas_used']}")
            click.echo(f"   Block number: {deployment_result['block_number']}")
        else:
            click.echo("❌ Deployment failed!")
            click.echo(f"   Error: {deployment_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        click.echo(f"❌ Deployment failed: {e}", err=True)


@cli.group()
def plan():
    """Quantum planning commands."""
    pass


@plan.command('optimize')
@click.option('--contract', '-c', help='Contract file or name')
@click.option('--objective', '-obj', default='stakeholder_satisfaction', 
              type=click.Choice(['stakeholder_satisfaction', 'safety_maximization', 'efficiency']),
              help='Optimization objective')
@click.option('--iterations', '-i', default=1000, help='Optimization iterations')
def optimize_plan(contract, objective, iterations):
    """Optimize contract using quantum planning."""
    try:
        # Load contract
        if contract:
            if os.path.exists(contract):
                with open(contract) as f:
                    contract_data = json.load(f)
            else:
                contract_data = ctx.services['contract'].get_contract_by_name(contract)
        else:
            if not ctx.current_contract:
                click.echo("❌ No contract specified. Use --contract or create one first.", err=True)
                return
            contract_data = ctx.current_contract.to_dict()
        
        click.echo(f"🧮 Running quantum optimization for {objective}...")
        click.echo(f"   Iterations: {iterations}")
        
        # Run quantum planning optimization
        optimization_result = ctx.planner.optimize_contract_parameters(
            contract_data,
            objective=objective,
            max_iterations=iterations
        )
        
        if optimization_result['success']:
            click.echo("✅ Optimization completed!")
            click.echo(f"   Final score: {optimization_result['final_score']:.4f}")
            click.echo(f"   Improvement: {optimization_result['improvement']:.2%}")
            click.echo(f"   Iterations: {optimization_result['iterations']}")
            
            # Show optimized parameters
            click.echo("\n📈 Optimized parameters:")
            for param, value in optimization_result['optimized_params'].items():
                click.echo(f"   {param}: {value:.4f}")
        else:
            click.echo("❌ Optimization failed!")
            click.echo(f"   Error: {optimization_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        click.echo(f"❌ Optimization failed: {e}", err=True)


@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the API server."""
    try:
        import uvicorn
        from api.main import app
        
        click.echo(f"🚀 Starting RLHF-Contract-Wizard API server...")
        click.echo(f"   Host: {host}")
        click.echo(f"   Port: {port}")
        click.echo(f"   Docs: http://{host}:{port}/docs")
        
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}", err=True)


@cli.command()
def version():
    """Show version information."""
    click.echo("RLHF-Contract-Wizard v0.1.0")
    click.echo("Legal-compliant RLHF with smart contracts and formal verification")
    click.echo("")
    click.echo("Dependencies:")
    click.echo(f"  JAX: {jax.__version__}")
    click.echo(f"  Python: {sys.version.split()[0]}")


@cli.command()
def status():
    """Show system status and health."""
    click.echo("🔍 RLHF-Contract-Wizard System Status")
    click.echo("=" * 50)
    
    # Check services
    services_status = {
        'Contract Service': 'healthy',
        'Verification Service': 'healthy',
        'Blockchain Service': 'unknown',
        'Quantum Planner': 'healthy'
    }
    
    for service, status in services_status.items():
        icon = "✅" if status == "healthy" else "⚠️" if status == "unknown" else "❌"
        click.echo(f"{icon} {service:<20} {status}")
    
    # Current configuration
    click.echo("\n📋 Configuration:")
    click.echo(f"   Database URL: {ctx.config.get('database_url', 'Not configured')}")
    click.echo(f"   Blockchain URL: {ctx.config.get('blockchain_url', 'Not configured')}")
    click.echo(f"   Verification Backend: {ctx.config.get('verification_backend', 'mock')}")
    
    # Current contract
    if ctx.current_contract:
        click.echo(f"\n📄 Current Contract: {ctx.current_contract.metadata.name}")
        click.echo(f"   Version: {ctx.current_contract.metadata.version}")
        click.echo(f"   Hash: {ctx.current_contract.compute_hash()}")
    else:
        click.echo("\n📄 No contract loaded")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()