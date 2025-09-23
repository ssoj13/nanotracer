# Build script for NanoTracer

# Clean previous builds
echo "Cleaning previous builds..."
cargo clean

# Build in release mode
echo "Building in release mode..."
cargo build --release

# Check if build was successful
if ($LASTEXITCODE -eq 0) {
    echo "Build successful!"
    
    # Run the application
    echo "Running NanoTracer..."
    cargo run --release
} else {
    echo "Build failed!"
    exit $LASTEXITCODE
}

# Pause to see output
Write-Host "Press any key to continue..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")