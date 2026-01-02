#!/usr/bin/env python3
"""
Architecture Consistency Validator

Scans the codebase for violations of ARCHITECTURE_RULES.md standards.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

class ArchitectureValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []

    def validate_all(self) -> Dict[str, List[str]]:
        """Run all validations and return violations by category"""
        results = {
            "logger_api": self.validate_logger_calls(),
            "error_handling": self.validate_error_handling(),
            "data_types": self.validate_data_types(),
        }
        return results

    def validate_logger_calls(self) -> List[str]:
        """Check for incorrect logger API usage"""
        violations = []

        # Patterns to detect incorrect usage
        patterns = [
            (r'self\.logger\.exception\(', "logger.exception() doesn't exist - use logger.error()"),
            (r'self\.logger\.debug\(f"', "logger.debug() requires component parameter: logger.debug(component, message)"),
            (r'self\.logger\.error\(f"', "logger.error() requires component parameter: logger.error(component, message, error)"),
        ]

        # Scan all Python files
        python_files = list(self.project_root.glob("src/**/*.py"))

        for file_path in python_files:
            try:
                content = file_path.read_text()
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, message in patterns:
                        if re.search(pattern, line):
                            violations.append(
                                f"{file_path.relative_to(self.project_root)}:{line_num} - {message}\n"
                                f"  Line: {line.strip()}"
                            )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return violations

    def validate_error_handling(self) -> List[str]:
        """Check for missing error handling patterns"""
        violations = []

        # Files that should have comprehensive error handling
        critical_files = [
            self.project_root / "src/crew/coordinator.py",
            self.project_root / "src/api/routes.py",
        ]

        for file_path in critical_files:
            if not file_path.exists():
                violations.append(f"{file_path.name} - File not found")
                continue

            content = file_path.read_text()

            # Check for crew.kickoff() calls without try/except
            kickoff_pattern = r'\.kickoff\(\)'
            try_pattern = r'try:'

            kickoff_matches = list(re.finditer(kickoff_pattern, content))

            for match in kickoff_matches:
                # Check if there's a try block within 5 lines before
                start = max(0, match.start() - 500)
                context = content[start:match.start()]

                if 'try:' not in context:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append(
                        f"{file_path.relative_to(self.project_root)}:{line_num} - "
                        f"crew.kickoff() should be in try/except block"
                    )

        return violations

    def validate_data_types(self) -> List[str]:
        """Check for data type mismatches"""
        violations = []

        coordinator_file = self.project_root / "src/crew/coordinator.py"

        if coordinator_file.exists():
            content = coordinator_file.read_text()

            # Check if character_arc conversion is present
            if 'isinstance(char_data[\'character_arc\'], list)' not in content:
                violations.append(
                    f"{coordinator_file.relative_to(self.project_root)} - "
                    f"Missing character_arc list→dict conversion in create_characters()"
                )

        return violations

    def print_report(self, results: Dict[str, List[str]]):
        """Print a formatted report of violations"""
        print("\n" + "="*80)
        print(" ARCHITECTURE CONSISTENCY VALIDATION REPORT")
        print("="*80 + "\n")

        total_violations = sum(len(v) for v in results.values())

        if total_violations == 0:
            print("✅ No violations found! Codebase follows all architecture rules.\n")
            return

        print(f"❌ Found {total_violations} violation(s)\n")

        for category, violations in results.items():
            if violations:
                print(f"\n{'─'*80}")
                print(f"  {category.upper().replace('_', ' ')} ({len(violations)} issues)")
                print(f"{'─'*80}\n")

                for i, violation in enumerate(violations, 1):
                    print(f"{i}. {violation}\n")

        print("="*80)
        print(f"\nTotal violations: {total_violations}")
        print("\nRefer to ARCHITECTURE_RULES.md for correct patterns.")
        print("="*80 + "\n")


def main():
    project_root = Path(__file__).parent.parent
    validator = ArchitectureValidator(project_root)

    print("🔍 Scanning codebase for architecture violations...")
    results = validator.validate_all()
    validator.print_report(results)

    # Exit with error code if violations found
    total_violations = sum(len(v) for v in results.values())
    exit(0 if total_violations == 0 else 1)


if __name__ == "__main__":
    main()
