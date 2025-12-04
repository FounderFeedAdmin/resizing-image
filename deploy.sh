#!/bin/bash

# Multi-Model Image Editor - Replicate Deployment Script

echo "🚀 Multi-Model Image Editor - Replicate Deployment"
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi
echo "✅ Docker is running"

# Function to check if logged in to Replicate
check_replicate_login() {
    echo "🔐 Checking Replicate login status..."
    
    # Check if we can list models (indicates we're logged in)
    if cog login --help > /dev/null 2>&1; then
        echo "✅ Cog login command available"
        return 0
    else
        echo "❌ Need to login to Replicate"
        return 1
    fi
}

# Function to login to Replicate
login_to_replicate() {
    echo "🔑 Logging in to Replicate..."
    echo "Please visit: https://replicate.com/account/api-tokens"
    echo "Copy your API token and paste it below:"
    
    if ! cog login; then
        echo "❌ Failed to login to Replicate"
        exit 1
    fi
    
    echo "✅ Successfully logged in to Replicate!"
}

# Check login status
if ! check_replicate_login; then
    login_to_replicate
fi
# Set the target model details
REPLICATE_USERNAME="founderfeed"
MODEL_NAME="image-resizer"
FULL_MODEL_NAME="r8.im/${REPLICATE_USERNAME}/${MODEL_NAME}"

echo "📝 Deploying as user: ${REPLICATE_USERNAME}"
echo "📦 Model name: ${MODEL_NAME}"
echo "🎯 Full model path: ${FULL_MODEL_NAME}"

echo ""
echo "🔨 Building and deploying model..."
echo "This may take a few minutes..."

# Deploy the model
if cog push "${FULL_MODEL_NAME}"; then
    echo ""
    echo "🎉 SUCCESS! Model deployed successfully!"
    echo ""
    echo "🔗 Your model is available at:"
    echo "   https://replicate.com/${REPLICATE_USERNAME}/${MODEL_NAME}"
    echo ""
    echo "📖 Usage instructions:"
    echo "   1. Visit the model page above"
    echo "   2. Upload one or more reference images"
    echo "   3. Enter your editing prompt"
    echo "   4. Select your preferred model (FLUX 2 Flex, Qwen, Nano Banana Pro, GPT Image Editor, or Seedream-4)"
    echo "   5. Optionally enable auto-upscaling"
    echo "   6. Click 'Run' to generate edited images"
    echo ""
    echo "✅ Deployment complete!"
    
    # Test the deployed model
    echo ""
    echo "🧪 Testing deployed model..."
    echo "Visit: https://replicate.com/${REPLICATE_USERNAME}/${MODEL_NAME}"
    
else
    echo ""
    echo "❌ DEPLOYMENT FAILED!"
    echo ""
    echo "Common issues and solutions:"
    echo "1. 🔑 Authentication: Make sure you're logged in with 'cog login'"
    echo "2. 📝 Model name: Ensure the model name is unique or you own it"
    echo "3. 🐳 Docker: Ensure Docker is running and has enough memory"
    echo "4. 🌐 Network: Check your internet connection"
    echo ""
    echo "💡 Try running the deployment command manually:"
    echo "   cog push ${FULL_MODEL_NAME}"
    exit 1
fi 