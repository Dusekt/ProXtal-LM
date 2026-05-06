#!/bin/bash
# Setup script for ProXtal-LM
# Creates symlinks to data files and sets up the environment

echo "🔧 Setting up ProXtal-LM..."

# Create symlinks to data files (modify paths as needed)
CRYSTALPRED_DIR="../ProXtal-LM_data"

if [ -d "$CRYSTALPRED_DIR" ]; then
    echo "📂 Creating symlinks to data files..."
    
    # Create data directory if it doesn't exist
    mkdir -p data
    
    # Link data files (adjust names as needed)
    if [ -f "$CRYSTALPRED_DIR/train_data_3d_cln" ]; then
        ln -sf "$CRYSTALPRED_DIR/train_data_3d_cln" data/train_data_3d_cln
        echo "  ✓ Linked training data"
    fi
    
    if [ -f "$CRYSTALPRED_DIR/valid_data_3d_cln" ]; then
        ln -sf "$CRYSTALPRED_DIR/valid_data_3d_cln" data/valid_data_3d_cln
        echo "  ✓ Linked validation data"
    fi
    
    if [ -f "$CRYSTALPRED_DIR/test_data_3d_cln" ]; then
        ln -sf "$CRYSTALPRED_DIR/test_data_3d_cln" data/test_data_3d_cln
        echo "  ✓ Linked test data"
    fi
else
    echo "⚠️  crystalpred directory not found at $CRYSTALPRED_DIR"
    echo "   You'll need to manually set data paths in the config"
fi

# Make scripts executable
echo "🔑 Making scripts executable..."
chmod +x scripts/*.py
chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Update data paths in proxtal_lm/config.py if needed"
echo "  2. Run: python scripts/train.py --config small"
echo "  3. Monitor training in checkpoints/training_metrics.csv"
