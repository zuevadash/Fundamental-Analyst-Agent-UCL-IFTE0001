[README.md](https://github.com/user-attachments/files/24986341/README.md)
# Fundamental Analyst Agent

**IFTE0001: AI Agents in Asset Management — Track A**

A comprehensive AI-powered fundamental analysis system that automates the complete equity research workflow, from data acquisition through LLM-generated investment memoranda.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Project Structure](#project-structure)
8. [Output Specifications](#output-specifications)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [Version History](#version-history)

---

## Overview

The Fundamental Analyst Agent implements a 9-phase analytical pipeline that simulates the role of a buy-side equity analyst. The system collects financial data, performs comprehensive validation, computes financial ratios, executes multiple valuation methodologies, and generates professional investment memoranda using Claude Sonnet 4.5.

### Coursework Deliverables

| Requirement | Implementation |
|-------------|----------------|
| Data ingestion pipeline (5 years) | Phase 1: Alpha Vantage integration with caching |
| Computed ratios (profitability, leverage, growth, efficiency) | Phase 3: 39 ratios across 5 categories |
| Basic intrinsic valuation (DCF or multiples) | Phases 5-7: DCF, DDM, and Multiples valuation |
| LLM-generated investment memo | Phase 9: Claude Sonnet 4.5 integration |
| Clean, reproducible code | Modular architecture with comprehensive documentation |

---

## Features

### Data Acquisition (Phase 1)
- 5-year financial statement collection (Income, Balance Sheet, Cash Flow)
- Company overview and market data retrieval
- Intelligent 5-hour caching with metadata tracking
- Data quality scoring and tier classification
- Accounting equation validation (Assets = Liabilities + Equity)
- Derived metrics calculation (FCF, EBITDA, Working Capital, Net Debt)
- Dividend history extraction with CAGR calculation
- Optional Yahoo Finance data supplementation

### Data Validation (Phase 2)
- Cross-statement reconciliation (Net Income, D&A consistency)
- Statistical outlier detection (IQR and Z-score methods)
- Sign convention validation
- Year-over-year and CAGR growth rate calculations
- Trend classification (Strong Growth to Strong Decline)
- Data standardization for downstream analysis

### Financial Ratio Analysis (Phase 3)
- 39 ratios across 5 categories:
  - Profitability (12 ratios): ROE, ROA, ROIC, margins
  - Leverage (8 ratios): D/E, D/A, interest coverage
  - Liquidity (5 ratios): Current, quick, cash ratios
  - Efficiency (7 ratios): Asset turnover, inventory days
  - Growth (7 ratios): Revenue, EPS, FCF, dividend CAGRs
- Benchmark-based assessment (Excellent to Critical)
- Time-series trend analysis
- Category-level and overall financial health scoring

### DuPont Analysis (Phase 4)
- 3-Factor decomposition: ROE = NPM x Asset Turnover x Equity Multiplier
- 5-Factor decomposition: Tax Burden x Interest Burden x Operating Margin x AT x EM
- Year-over-year variance attribution with exact reconciliation
- ROE quality and sustainability assessment
- Primary driver identification (Profitability, Efficiency, or Leverage)

### DCF Valuation (Phase 5)
- Historical FCF analysis and quality assessment
- Multi-method growth rate derivation (Historical CAGR, Sustainable, Analyst)
- WACC calculation via CAPM (Risk-free rate, Beta, Equity Risk Premium)
- 5-year explicit FCF projection
- Gordon Growth terminal value methodology
- Sensitivity analysis (growth rate vs. discount rate matrix)
- Scenario analysis (Bear, Base, Bull cases)
- Intrinsic value per share with market comparison

### DDM Valuation (Phase 6)
- DDM applicability assessment (dividend history, payout stability)
- Historical dividend analysis and quality scoring
- Dividend growth rate derivation (CAGR, Sustainable, Earnings-based)
- Cost of Equity calculation via CAPM
- 5-year dividend projection with terminal value
- Sensitivity and scenario analysis
- DCF vs. DDM reconciliation

### Multiples Valuation (Phase 7)
- 6 valuation multiples: P/E, P/B, P/S, P/FCF, EV/EBITDA, EV/Revenue
- Historical average comparisons (5-year trailing)
- Premium/discount assessment vs. historical averages
- Implied fair value calculation from each multiple
- Weighted composite valuation signal

### Accuracy Verification (Phase 8)
- 43 verification checks across 6 categories:
  - Input integrity validation
  - Calculation accuracy verification
  - Cross-phase consistency checks
  - Valuation model consistency
  - Range validity assessment
  - Methodology compliance verification
- Confidence scoring (0-100%)
- Reliability assessment for memo generation

### Investment Memo Generation (Phase 9)
- Claude Sonnet 4.5 LLM integration
- Professional institutional-grade memo format
- Multi-format output (JSON, Markdown, PDF)
- Data-driven recommendation synthesis
- Risk assessment and catalyst identification
- Consensus valuation with model weighting

---

## Requirements

### System Requirements
- Python 3.10 or higher
- 4 GB RAM minimum
- Internet connection for API calls

### API Keys Required
- Alpha Vantage API Key (free tier: 25 calls/day)
- Anthropic API Key (for Phase 9 memo generation)

### Python Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
reportlab>=4.0.0
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fundamental-analyst-agent
```

### 2. Create Virtual Environment

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
# Linux/macOS
export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
export ANTHROPIC_API_KEY=your_anthropic_key

# Windows (Command Prompt)
set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
set ANTHROPIC_API_KEY=your_anthropic_key

# Windows (PowerShell)
$env:ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
$env:ANTHROPIC_API_KEY="your_anthropic_key"
```

### Obtaining API Keys

**Alpha Vantage (Free)**
1. Visit https://www.alphavantage.co/support/#api-key
2. Register for a free API key
3. Free tier limits: 5 calls/minute, 25 calls/day

**Anthropic**
1. Visit https://console.anthropic.com/
2. Create an account and generate an API key
3. Required for Phase 9 memo generation

---

## Usage

### Basic Usage

```bash
# Analyze Apple (default ticker)
python run_demo.py

# Analyze a specific company
python run_demo.py MSFT

# Analyze with custom API key
python run_demo.py AAPL --api-key your_anthropic_key
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `ticker` | Stock ticker symbol (default: AAPL) |
| `--refresh` | Force data refresh, bypass cache |
| `--quiet` | Suppress detailed output, show summary only |
| `--phase1-only` | Run Phase 1 only (data acquisition) |
| `--phase2-only` | Run Phases 1-2 (through validation) |
| `--phase3-only` | Run Phases 1-3 (through ratio analysis) |
| `--phase4-only` | Run Phases 1-4 (through DuPont analysis) |
| `--phase5-only` | Run Phases 1-5 (through DCF valuation) |
| `--phase6-only` | Run Phases 1-6 (through DDM valuation) |
| `--phase7-only` | Run Phases 1-7 (through multiples valuation) |
| `--phase8-only` | Run Phases 1-8 (skip memo generation) |
| `--api-key` | Anthropic API key for Phase 9 |
| `--no-supplement` | Disable Yahoo Finance supplementation |

### Examples

```bash
# Full analysis with detailed output
python run_demo.py MSFT

# Quick analysis with summary output
python run_demo.py GOOGL --quiet

# Refresh cached data
python run_demo.py AAPL --refresh

# Run valuation phases only (skip memo)
python run_demo.py NVDA --phase8-only

# Run ratio analysis only
python run_demo.py AMZN --phase3-only
```

---

## Pipeline Architecture

```
                        FUNDAMENTAL ANALYST AGENT
                         9-Phase Analysis Pipeline

+------------+    +------------+    +------------+    +------------+
|  PHASE 1   |--->|  PHASE 2   |--->|  PHASE 3   |--->|  PHASE 4   |
|    Data    |    |    Data    |    |   Ratio    |    |   DuPont   |
| Acquisition|    | Validation |    |  Analysis  |    |  Analysis  |
+------------+    +------------+    +------------+    +------------+
      |                                                      |
      |           +------------------------------------------+
      |           |
      v           v
+------------+    +------------+    +------------+    +------------+
|  PHASE 5   |--->|  PHASE 6   |--->|  PHASE 7   |--->|  PHASE 8   |
|    DCF     |    |    DDM     |    | Multiples  |    |  Accuracy  |
| Valuation  |    | Valuation  |    | Valuation  |    |   Check    |
+------------+    +------------+    +------------+    +------------+
                                                            |
                                                            v
                                                     +------------+
                                                     |  PHASE 9   |
                                                     |  LLM Memo  |
                                                     | Generation |
                                                     +------------+
                                                            |
                                                            v
                                                     +------------+
                                                     |   OUTPUT   |
                                                     | JSON/MD/PDF|
                                                     +------------+
```

### Data Flow

| Phase | Input | Output | Key Metrics |
|-------|-------|--------|-------------|
| 1 | Ticker Symbol | CollectionResult | 5-year statements, derived metrics |
| 2 | CollectionResult | ValidatedData | Reconciliation scores, outlier flags |
| 3 | CollectionResult | RatioAnalysisResult | 39 ratios, category scores |
| 4 | CollectionResult | DuPontAnalysisResult | 3/5-factor decomposition, ROE quality |
| 5 | CollectionResult | DCFValuationResult | Intrinsic value, WACC, scenarios |
| 6 | CollectionResult, DCF | DDMValuationResult | DDM value, applicability |
| 7 | CollectionResult, DCF, DDM | MultiplesValuationResult | Implied fair value |
| 8 | All prior results | AccuracyCheckResult | 43 checks, confidence score |
| 9 | All prior results | InvestmentMemoResult | JSON, Markdown, PDF memo |

---

## Project Structure

```
fundamental-analyst-agent/
|
|-- run_demo.py              # Main entry point
|-- requirements.txt         # Python dependencies
|-- README.md                # Documentation
|
|-- src/
|   |-- __init__.py          # Package exports and version info
|   |-- config.py            # Configuration constants
|   |                        #   - Field mappings
|   |                        #   - Validation thresholds
|   |                        #   - Ratio benchmarks
|   |                        #   - Sign conventions
|   |
|   |-- data_collector.py    # Phase 1: Data Acquisition
|   |                        #   - Alpha Vantage API integration
|   |                        #   - Caching system
|   |                        #   - Quality metrics
|   |                        #   - Derived calculations
|   |
|   |-- data_validator.py    # Phase 2: Validation
|   |                        #   - Cross-statement reconciliation
|   |                        #   - Outlier detection
|   |                        #   - Growth analysis
|   |
|   |-- ratio_analyzer.py    # Phase 3: Ratio Analysis
|   |                        #   - 39 financial ratios
|   |                        #   - Benchmark assessment
|   |                        #   - Trend analysis
|   |
|   |-- dupont_analyzer.py   # Phase 4: DuPont Analysis
|   |                        #   - 3-factor decomposition
|   |                        #   - 5-factor decomposition
|   |                        #   - Variance attribution
|   |
|   |-- dcf_valuator.py      # Phase 5: DCF Valuation
|   |                        #   - FCF projections
|   |                        #   - WACC calculation
|   |                        #   - Sensitivity analysis
|   |
|   |-- ddm_valuator.py      # Phase 6: DDM Valuation
|   |                        #   - Dividend projections
|   |                        #   - Applicability assessment
|   |                        #   - DCF reconciliation
|   |
|   |-- multiples_valuator.py # Phase 7: Multiples Valuation
|   |                        #   - 6 valuation multiples
|   |                        #   - Historical comparisons
|   |                        #   - Composite scoring
|   |
|   |-- accuracy_checker.py  # Phase 8: Verification
|   |                        #   - 43 accuracy checks
|   |                        #   - Confidence scoring
|   |                        #   - Reliability assessment
|   |
|   |-- memo_generator.py    # Phase 9: LLM Memo Generation
|                            #   - Claude Sonnet 4.5 integration
|                            #   - PDF generation
|                            #   - Multi-format output
|
|-- cache/                   # Cached API responses (auto-created)
|   |-- {TICKER}_*.json
|
|-- outputs/                 # Analysis outputs (auto-created)
    |-- {TICKER}_collection.json
    |-- {TICKER}_validation.json
    |-- {TICKER}_ratios.json
    |-- {TICKER}_dupont.json
    |-- {TICKER}_dcf.json
    |-- {TICKER}_ddm.json
    |-- {TICKER}_multiples.json
    |-- {TICKER}_accuracy.json
    |-- {TICKER}_investment_memo.json
    |-- {TICKER}_investment_memo.md
    |-- {TICKER}_investment_memo.pdf
```

---

## Output Specifications

### Data Quality Tiers

| Tier | Completeness | Description |
|------|--------------|-------------|
| EXCELLENT | >= 95% | Full data coverage, all critical fields present |
| GOOD | >= 80% | Minor gaps, suitable for full analysis |
| ACCEPTABLE | >= 60% | Some gaps, limited analysis possible |
| POOR | < 60% | Significant gaps, unreliable analysis |

### Valuation Signals

| Signal | Upside Range | Description |
|--------|--------------|-------------|
| STRONG BUY | >= +25% | Significantly undervalued |
| BUY | +10% to +25% | Moderately undervalued |
| HOLD | -10% to +10% | Fairly valued |
| SELL | -25% to -10% | Moderately overvalued |
| STRONG SELL | <= -25% | Significantly overvalued |

### Accuracy Check Categories

| Category | Weight | Description |
|----------|--------|-------------|
| Input Integrity | 15% | Source data validation |
| Calculation Accuracy | 25% | Re-verification of computations |
| Cross-Phase Consistency | 20% | Data alignment across phases |
| Valuation Consistency | 20% | Model input validation |
| Range Validity | 10% | Reasonable bounds checking |
| Methodology Compliance | 10% | Implementation verification |

### Confidence Levels

| Level | Score | Reliability |
|-------|-------|-------------|
| VERY HIGH | >= 95% | Analysis highly reliable |
| HIGH | >= 85% | Analysis reliable |
| ACCEPTABLE | >= 70% | Analysis usable with caveats |
| LOW | >= 50% | Analysis questionable |
| VERY LOW | < 50% | Analysis unreliable |

---

## Configuration

### Alpha Vantage Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rate Limit | 12 seconds | Delay between API calls |
| Cache Expiry | 5 hours | Time before data refresh |
| Daily Limit | 25 calls | Free tier API limit |
| Minute Limit | 5 calls | Free tier rate limit |

### DCF Valuation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Risk-Free Rate | 4.3% | 10-Year US Treasury |
| Equity Risk Premium | 5.5% | Historical average |
| Terminal Growth | 2.5% | Long-term GDP proxy |
| Projection Years | 5 | Explicit forecast period |
| Min WACC | 6.0% | Floor for discount rate |
| Max WACC | 20.0% | Cap for discount rate |

### DDM Valuation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Dividend Years | 3 | Required history for applicability |
| Min Payout Ratio | 10% | Floor for DDM validity |
| Max Payout Ratio | 95% | Cap for DDM validity |
| Max Dividend CV | 50% | Stability threshold |

### Ratio Benchmarks (Profitability)

| Ratio | Excellent | Good | Acceptable | Weak |
|-------|-----------|------|------------|------|
| Gross Margin | > 50% | > 35% | > 25% | > 15% |
| Operating Margin | > 25% | > 15% | > 8% | > 3% |
| Net Profit Margin | > 20% | > 10% | > 5% | > 2% |
| ROE | > 20% | > 15% | > 10% | > 5% |
| ROA | > 10% | > 7% | > 4% | > 2% |

---

## API Reference

### Core Classes

#### DataCollector (Phase 1)
```python
from src.data_collector import DataCollector, CollectionResult

collector = DataCollector(enable_supplementation=True)
result: CollectionResult = collector.collect("AAPL", force_refresh=False)
```

#### Phase2Validator (Phase 2)
```python
from src.data_validator import Phase2Validator, ValidatedData

validator = Phase2Validator()
validated: ValidatedData = validator.validate(collection_result)
```

#### Phase3Analyzer (Phase 3)
```python
from src.ratio_analyzer import Phase3Analyzer, RatioAnalysisResult

analyzer = Phase3Analyzer()
ratios: RatioAnalysisResult = analyzer.analyze(collection_result)
```

#### Phase4Analyzer (Phase 4)
```python
from src.dupont_analyzer import Phase4Analyzer, DuPontAnalysisResult

dupont = Phase4Analyzer()
result: DuPontAnalysisResult = dupont.analyze(collection_result)
```

#### Phase5Valuator (Phase 5)
```python
from src.dcf_valuator import Phase5Valuator, DCFValuationResult

valuator = Phase5Valuator()
dcf: DCFValuationResult = valuator.value(collection_result)
```

#### Phase6Valuator (Phase 6)
```python
from src.ddm_valuator import Phase6Valuator, DDMValuationResult

valuator = Phase6Valuator()
ddm: DDMValuationResult = valuator.value(collection_result, dcf_result)
```

#### Phase7Valuator (Phase 7)
```python
from src.multiples_valuator import Phase7Valuator, MultiplesValuationResult

valuator = Phase7Valuator()
multiples: MultiplesValuationResult = valuator.value(
    collection_result, dcf_result, ddm_result
)
```

#### Phase8AccuracyChecker (Phase 8)
```python
from src.accuracy_checker import Phase8AccuracyChecker, AccuracyCheckResult

checker = Phase8AccuracyChecker()
accuracy: AccuracyCheckResult = checker.check(
    collection_result, validated_data, ratio_result,
    dupont_result, dcf_result, ddm_result, multiples_result
)
```

#### Phase9MemoGenerator (Phase 9)
```python
from src.memo_generator import Phase9MemoGenerator, InvestmentMemoResult

generator = Phase9MemoGenerator(api_key="your_key")
memo: InvestmentMemoResult = generator.generate(
    collection_result, validated_data, ratio_result,
    dupont_result, dcf_result, ddm_result, multiples_result, accuracy_result
)
```

---

## Troubleshooting

### Common Issues

**API Key Not Found**
```
Error: ALPHA_VANTAGE_API_KEY environment variable not set
```
Solution: Set the environment variable as described in the Installation section.

**Rate Limit Exceeded**
```
Error: API rate limit reached
```
Solution: Wait for the rate limit to reset (1 minute for per-minute limits, 24 hours for daily limits) or use cached data.

**DDM Not Applicable**
```
DDM Status: Not Applicable
```
This is expected for companies that do not pay dividends or have unstable dividend histories. The system will exclude DDM from the consensus valuation.

**Memo Generation Failed**
```
Status: FAILED
Error: API request failed
```
Solution: Verify your Anthropic API key is valid and has available credits.

### Cache Management

```bash
# Clear cache for specific ticker
rm cache/AAPL_*.json

# Clear all cached data
rm -rf cache/*.json

# Force refresh without clearing cache
python run_demo.py AAPL --refresh
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.0.0 | 2026-01 | Added Phase 9 LLM memo generation |
| 3.3.0 | 2026-01 | Added Phase 8 accuracy verification |
| 3.2.0 | 2026-01 | Added Phase 7 multiples valuation |
| 3.1.0 | 2026-01 | Added Phase 6 DDM valuation |
| 3.0.0 | 2025-12 | Added Phase 5 DCF valuation |
| 2.0.0 | 2025-12 | Added Phases 3-4 (ratios, DuPont) |
| 1.0.0 | 2025-11 | Initial release (Phases 1-2) |

---

## License

This project is developed as coursework for IFTE0001: AI Agents in Asset Management at UCL IFT.

---

## Acknowledgments

- Alpha Vantage for financial data API
- Anthropic for Claude LLM API
- UCL IFT for coursework framework
