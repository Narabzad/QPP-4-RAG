# GitHub Repository Setup Guide

This guide helps you prepare and push the repository to GitHub.

## Pre-Push Checklist

### Documentation Files ✅
- [x] `README.md` - Main repository documentation
- [x] `REPRODUCIBILITY.md` - Detailed reproduction instructions
- [x] `FILES_INVENTORY.md` - Complete file listing
- [x] `QUICKSTART.md` - Quick start guide
- [x] `PAPER_INFO.md` - Paper information and key findings
- [x] `.gitignore` - Git ignore file

### Step 1: Query Generation ✅
- [ ] Verify `querygym/queries/` contains 31 query files
- [ ] Include `querygym/generate_query_files.py` script

### Step 2: RAG Execution ✅
- [ ] Verify `querygym/rag_prepared/` contains prepared files (31 × 2 = 62 files)
- [ ] Verify `querygym/rag_results/` contains RAG results (31 × 2 = 62 files)
- [ ] Include `querygym/run_RAG_on_prepared_files.py` script
- [ ] Include `scripts/NO_RRF_convert_to_ragnarok_format.py` script
- [ ] **Exclude RRF results** (as per paper requirements)

### Step 3: Nuggetizer Evaluation ✅
- [ ] Verify `querygym/rag_nuggetized_eval/` contains evaluation results
- [ ] Include `querygym/run_rag_nuggetizer.py` script

### Step 4: QPP Predictions ✅
- [ ] Verify `querygym/qpp/` contains:
  - 31 pre-retrieval QPP CSV files
  - 62 post-retrieval QPP CSV files (31 × 2 retrieval methods)
- [ ] Verify `querygym/qpp/bert_qpp_results/` contains BERT-QPP predictions
- [ ] Verify `querygym/qpp/QSDQPP/` contains QSDQPP predictions (31 files)
- [ ] Include QPP computation scripts:
  - `querygym/qpp/run_pre_retrieval_verbose.py`
  - `querygym/qpp/run_qpp_querygym.py`
  - `querygym/run_bert_qpp.py`
- [ ] Include QPP implementations:
  - `QPP4CS/` directory
  - `BERTQPP/` directory

## GitHub Repository Setup

### 1. Initialize Git Repository (if not already done)

```bash
cd /future/u/negara/home/set_based_QPP
git init
```

### 2. Add Remote Repository

```bash
git remote add origin https://github.com/Narabzad/QPP-4-RAG.git
```

Or if using SSH:
```bash
git remote add origin git@github.com:Narabzad/QPP-4-RAG.git
```

### 3. Stage Files

**Important**: Review what you're committing. The repository should include:

```bash
# Add documentation
git add README.md REPRODUCIBILITY.md FILES_INVENTORY.md QUICKSTART.md PAPER_INFO.md .gitignore

# Add query files
git add querygym/queries/*.txt

# Add scripts
git add querygym/*.py
git add scripts/NO_RRF_convert_to_ragnarok_format.py
git add querygym/qpp/*.py
git add querygym/run_*.py

# Add QPP implementations
git add QPP4CS/
git add BERTQPP/

# Add results (be careful with large files)
# Option 1: Add all results
git add querygym/rag_prepared/
git add querygym/rag_results/
git add querygym/rag_nuggetized_eval/
git add querygym/qpp/*.csv
git add querygym/qpp/bert_qpp_results/
git add querygym/qpp/QSDQPP/*_predicted_ndcg.txt

# Option 2: Use Git LFS for large files (recommended)
# First install git-lfs, then:
git lfs install
git lfs track "*.json"
git lfs track "*.jsonl"
git add .gitattributes
```

### 4. Commit Files

```bash
git commit -m "Initial commit: QPP-4-RAG reproducibility package

- Query generation: 31 query variant files
- RAG execution: Prepared files and results for 31 variants × 2 retrieval methods
- Nuggetizer evaluation: Complete evaluation results
- QPP predictions: Pre-retrieval, post-retrieval, BERT-QPP, and QSDQPP
- Complete documentation for reproducibility"
```

### 5. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

## Handling Large Files

If you encounter issues with large files:

### Option 1: Use Git LFS

```bash
# Install git-lfs
git lfs install

# Track large file types
git lfs track "*.json"
git lfs track "*.jsonl"
git lfs track "*.csv"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add .
git commit -m "Add Git LFS tracking for large files"
git push
```

### Option 2: Exclude Large Result Files

If files are too large, you can:
1. Document that results can be regenerated
2. Provide a script to download results separately
3. Use external storage (Zenodo, Google Drive, etc.) and link in README

### Option 3: Compress Results

```bash
# Create compressed archives
tar -czf rag_results.tar.gz querygym/rag_results/
tar -czf rag_prepared.tar.gz querygym/rag_prepared/
tar -czf nuggetizer_eval.tar.gz querygym/rag_nuggetized_eval/

# Add to .gitignore
echo "*.tar.gz" >> .gitignore

# Upload to external storage and link in README
```

## Verification After Push

1. **Check repository on GitHub**: Verify all files are present
2. **Test cloning**: Clone the repository in a fresh location to verify
3. **Check file sizes**: Ensure large files are handled appropriately
4. **Verify links**: Check all documentation links work

## Post-Push Tasks

1. **Add repository description**: "Query Performance Prediction for Retrieval-Augmented Generation - SIGIR 2026"
2. **Add topics**: `qpp`, `rag`, `information-retrieval`, `query-performance-prediction`, `sigir2026`
3. **Add license**: Choose and add appropriate license file
4. **Create releases**: Tag major versions if needed
5. **Enable issues**: Allow users to report issues and ask questions

## Repository Structure on GitHub

The final repository should have this structure:

```
QPP-4-RAG/
├── README.md
├── REPRODUCIBILITY.md
├── FILES_INVENTORY.md
├── QUICKSTART.md
├── PAPER_INFO.md
├── .gitignore
├── .gitattributes (if using Git LFS)
├── querygym/
│   ├── queries/ (31 files)
│   ├── rag_prepared/ (62 files)
│   ├── rag_results/ (62 files)
│   ├── rag_nuggetized_eval/ (124 files)
│   ├── qpp/ (93 CSV files)
│   │   └── bert_qpp_results/ (2 files)
│   │   └── QSDQPP/ (31 files)
│   └── [scripts]
├── QPP4CS/
├── BERTQPP/
└── scripts/
```

## Troubleshooting

### Issue: Files too large for GitHub
**Solution**: Use Git LFS or external storage

### Issue: Authentication errors
**Solution**: 
- Use personal access token instead of password
- Or set up SSH keys

### Issue: Push rejected
**Solution**: 
- Pull first: `git pull origin main --rebase`
- Resolve conflicts
- Push again

## Next Steps After Push

1. **Share repository**: Update paper with repository URL
2. **Create DOI**: Use Zenodo to create a DOI for the repository
3. **Update paper**: Add repository link to paper
4. **Monitor issues**: Respond to questions and issues

## Contact

If you encounter issues, check:
- GitHub documentation: https://docs.github.com
- Git LFS documentation: https://git-lfs.github.com
- Repository issues: Open an issue on GitHub
