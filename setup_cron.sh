#!/bin/bash

# Setup script for quantum batch processing with cron integration
# This script configures the batch worker to run on a schedule

echo "Setting up Quantum Batch Processing with Cron Integration"

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/quantum_batch_worker.py"

# Set environment variables for cron - retrieve from keychain
GROQ_API_KEY=$(security find-generic-password -s "groq API key" -w 2>/dev/null)

echo "Worker script location: $WORKER_SCRIPT"

# Create a cron environment file
cat > "$SCRIPT_DIR/cron_env.sh" << EOF
#!/bin/bash
# Retrieve API key from macOS keychain
GROQ_API_KEY=\$(security find-generic-password -s "groq API key" -w 2>/dev/null)
export GROQ_API_KEY="\$GROQ_API_KEY"
export PATH="/usr/local/bin:/usr/bin:/bin:\$PATH"
cd "$SCRIPT_DIR"
EOF

chmod +x "$SCRIPT_DIR/cron_env.sh"

# Create the cron job entry
CRON_JOB="*/5 * * * * $SCRIPT_DIR/cron_env.sh && cd $SCRIPT_DIR && python3 $WORKER_SCRIPT --check-once 2>&1 | logger -t quantum-batch"

echo "Proposed cron job:"
echo "$CRON_JOB"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "quantum-batch"; then
    echo "Quantum batch cron job already exists. Current crontab:"
    crontab -l | grep quantum-batch
else
    echo "Would you like to install this cron job? (y/n)"
    read -p "Install cron job? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Add the cron job
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        echo "Cron job installed successfully!"
        echo "The batch worker will check for jobs every 5 minutes."
    else
        echo "Cron job not installed. You can manually add it later:"
        echo "$CRON_JOB"
    fi
fi

echo ""
echo "To manually start the batch worker:"
echo "cd $SCRIPT_DIR && python3 $WORKER_SCRIPT"
echo ""
echo "To remove the cron job:"
echo "crontab -e  # then delete the line containing 'quantum-batch'"
echo ""
echo "To view cron logs:"
echo "tail -f /var/log/system.log | grep quantum-batch"