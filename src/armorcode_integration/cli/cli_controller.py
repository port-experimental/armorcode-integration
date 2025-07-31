"""
Enhanced CLI Controller for ArmorCode-Port Integration

This module provides enhanced command-line interface functionality with support for:
- Selective step execution
- Advanced filtering options
- Configurable batch processing
- Retry mechanisms
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CLIConfig:
    """Configuration class containing all CLI options and their values."""
    steps: List[str]
    after_key: Optional[int]
    finding_filters_path: Optional[str]
    batch_size: int
    retry_attempts: int
    dry_run: bool


class CLIController:
    """Enhanced CLI controller with validation and configuration management."""
    
    # Available steps that can be executed
    AVAILABLE_STEPS = ["product", "subproduct", "finding"]
    
    # Default configuration values
    DEFAULT_BATCH_SIZE = 20
    DEFAULT_RETRY_ATTEMPTS = 3
    MAX_BATCH_SIZE = 20
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Creates and configures the argument parser."""
        parser = argparse.ArgumentParser(
            description="Enhanced ArmorCode to Port Integration Script with selective execution and bulk operations.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
        Examples:
        armorcode-integration                                    # Run all steps with default settings
        armorcode-integration --steps product,subproduct        # Run only product and subproduct steps
        armorcode-integration --steps finding --after-key 12345 # Run findings step with afterKey filter
        armorcode-integration --finding-filters filters.json    # Apply JSON filters to findings
        armorcode-integration --batch-size 10 --retry-attempts 5 # Custom batch size and retry settings
        armorcode-integration --dry-run                         # Preview changes without executing
            """
        )
        
        parser.add_argument(
            "--steps",
            type=str,
            help=f"Comma-separated list of steps to execute. Available steps: {', '.join(self.AVAILABLE_STEPS)}. "
                 f"If not specified, all steps will be executed."
        )
        
        parser.add_argument(
            "--after-key",
            type=int,
            help="Integer value to use as afterKey parameter for ArmorCode findings API pagination. "
                 "Only findings created after this key will be fetched."
        )
        
        parser.add_argument(
            "--finding-filters",
            type=str,
            metavar="FILE",
            help="Path to JSON file containing FindingFiltersRequestDto format filters for ArmorCode findings API. "
                 "Can be combined with --after-key parameter."
        )
        
        parser.add_argument(
            "--batch-size",
            type=int,
            default=self.DEFAULT_BATCH_SIZE,
            help=f"Number of entities to process in each batch for bulk operations. "
                 f"Maximum allowed: {self.MAX_BATCH_SIZE}. Default: {self.DEFAULT_BATCH_SIZE}"
        )
        
        parser.add_argument(
            "--retry-attempts",
            type=int,
            default=self.DEFAULT_RETRY_ATTEMPTS,
            help=f"Number of retry attempts for failed API calls. "
                 f"Set to 0 to disable retries. Default: {self.DEFAULT_RETRY_ATTEMPTS}"
        )
        
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run the script without making any changes to Port. Useful for testing and validation."
        )
        
        return parser
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> CLIConfig:
        """
        Parses command line arguments and returns a validated configuration.
        
        Args:
            args: Optional list of arguments to parse. If None, uses sys.argv.
            
        Returns:
            CLIConfig: Validated configuration object
            
        Raises:
            SystemExit: If validation fails or help is requested
        """
        parsed_args = self.parser.parse_args(args)
        
        # Parse steps
        if parsed_args.steps:
            steps = [step.strip() for step in parsed_args.steps.split(",")]
        else:
            steps = self.AVAILABLE_STEPS.copy()
        
        # Create configuration
        config = CLIConfig(
            steps=steps,
            after_key=parsed_args.after_key,
            finding_filters_path=parsed_args.finding_filters,
            batch_size=parsed_args.batch_size,
            retry_attempts=parsed_args.retry_attempts,
            dry_run=parsed_args.dry_run
        )
        
        # Validate configuration
        self.validate_configuration(config)
        
        return config
    
    def validate_configuration(self, config: CLIConfig) -> None:
        """
        Validates the configuration and raises SystemExit with error message if invalid.
        
        Args:
            config: Configuration to validate
            
        Raises:
            SystemExit: If validation fails
        """
        errors = []
        
        # Validate steps
        invalid_steps = [step for step in config.steps if step not in self.AVAILABLE_STEPS]
        if invalid_steps:
            errors.append(
                f"Invalid step names: {', '.join(invalid_steps)}. "
                f"Available steps: {', '.join(self.AVAILABLE_STEPS)}"
            )
        
        if not config.steps:
            errors.append("At least one step must be specified")
        
        # Validate batch size
        if config.batch_size <= 0:
            errors.append("Batch size must be greater than 0")
        elif config.batch_size > self.MAX_BATCH_SIZE:
            errors.append(f"Batch size cannot exceed {self.MAX_BATCH_SIZE}")
        
        # Validate retry attempts
        if config.retry_attempts < 0:
            errors.append("Retry attempts cannot be negative")
        
        # Validate after key
        if config.after_key is not None and config.after_key < 0:
            errors.append("After key must be a non-negative integer")
        
        # Validate finding filters file
        if config.finding_filters_path:
            if not self._validate_filters_file(config.finding_filters_path):
                errors.append(f"Finding filters file does not exist or is not readable: {config.finding_filters_path}")
        
        # If there are errors, print them and exit
        if errors:
            self.parser.error("\n".join(f"Error: {error}" for error in errors))
    
    def _validate_filters_file(self, file_path: str) -> bool:
        """
        Validates that the filters file exists and is readable.
        
        Args:
            file_path: Path to the filters file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            if not path.is_file():
                return False
            if not os.access(path, os.R_OK):
                return False
            
            # Try to parse as JSON to ensure it's valid
            with open(path, 'r') as f:
                json.load(f)
            return True
        except (OSError, json.JSONDecodeError, PermissionError):
            return False
    
    def get_available_steps(self) -> List[str]:
        """
        Returns the list of available steps.
        
        Returns:
            List[str]: Available step names
        """
        return self.AVAILABLE_STEPS.copy()
    
    def print_configuration_summary(self, config: CLIConfig) -> None:
        """
        Prints a summary of the current configuration.
        
        Args:
            config: Configuration to summarize
        """
        print("Configuration Summary:")
        print(f"  Steps to execute: {', '.join(config.steps)}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Retry attempts: {config.retry_attempts}")
        print(f"  Dry run: {config.dry_run}")
        
        if config.after_key is not None:
            print(f"  After key: {config.after_key}")
        
        if config.finding_filters_path:
            print(f"  Finding filters file: {config.finding_filters_path}")
        
        print()