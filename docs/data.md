# Banner Animator Generator - Complete Developer Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Pipeline](#architecture--pipeline)
3. [Setup & Installation](#setup--installation)
4. [Core Components](#core-components)
5. [API Integration Guide](#api-integration-guide)
6. [Adding Input Fields](#adding-input-fields)
7. [Writing Custom Pipelines](#writing-custom-pipelines)
8. [Model Integration](#model-integration)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Customization](#advanced-customization)

---

## System Overview

The Banner Animator Generator is a complete AI-powered video pipeline that transforms text prompts into animated banner videos with optional audio. The system uses multiple AI models orchestrated through a single prediction interface.

### Key Features

- **Text-to-Image Generation**: Creates high-quality banner images using Ideogram v3 or Flux Kontext Pro
- **Image-to-Video Animation**: Animates static images using SeDance-1-Pro
- **Audio Generation**: Adds contextual audio using MMAudio v2
- **Style Customization**: Multiple visual styles, lighting, and camera movements
- **Reference Image Support**: Style transfer using reference images
- **Intelligent Parameter Selection**: Auto-selects optimal settings using GPT-4

---

## Architecture & Pipeline

### High-Level Flow

```
User Input → Parameter Selection → Prompt Enhancement → Image Generation → Video Animation → Audio Addition → Final Output
```

### Detailed Pipeline Steps

1. **Input Processing & Parameter Selection**

   - Analyze user prompt
   - Auto-select video style, lighting, and camera movement (if set to "auto")
   - Validate input parameters

2. **Prompt Enhancement**

   - Transform user input into professional commercial photography descriptions
   - Handle text detection and placement
   - Apply proven quality templates

3. **Image Generation**

   - **Without Reference**: Use Ideogram v3 for high-quality banner images
   - **With Reference**: Use Flux Kontext Pro for style-consistent generation

4. **Video Animation**

   - Generate video prompt using structured AI output
   - Animate image using SeDance-1-Pro I2V model
   - Apply selected camera movements and effects

5. **Audio Generation (Optional)**

   - Generate contextual audio prompts
   - Add audio using MMAudio v2 model
   - Sync audio with video content

6. **Output Processing**
   - Download and return final video file
   - Handle error cases and fallbacks

---

## Setup & Installation

### Prerequisites

```bash
# Python 3.8+
# Required system packages
pip install cog replicate openai python-dotenv cloudinary langfuse pydantic pillow requests
```

### Environment Variables

Create a `.env` file (keep it local, never commit):

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
REPLICATE_API_TOKEN=your_replicate_token_here

# Cloudinary Configuration
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Optional: Langfuse Tracking
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
```

### Cog Configuration

Create `cog.yaml`:

```yaml
build:
  python_version: "3.11"
  system_packages:
    - "libgl1-mesa-glx"
    - "libglib2.0-0"
  python_packages:
    - "replicate"
    - "openai"
    - "python-dotenv"
    - "cloudinary"
    - "langfuse"
    - "pydantic"
    - "pillow"
    - "requests"

predict: "predict.py:Predictor"
```

---

## Core Components

### 1. Predictor Class Structure

```python
class Predictor(BasePredictor):
    def setup(self):
        """Initialize once when container starts"""
        # Create output directories
        # Configure Cloudinary
        # Initialize services

    def predict(self, **inputs) -> Path:
        """Main prediction pipeline"""
        # Process inputs
        # Execute pipeline steps
        # Return final video file
```

### 2. Input Parameters

```python
# Text input
prompt: str = Input(description="Describe what you want to create")

# Style parameters
video_style: str = Input(choices=["auto", "3d_render", "cartoon", ...])
lighting: str = Input(choices=["auto", "studio", "neon", ...])
camera_movement: str = Input(choices=["auto", "move_left", "push_in", ...])

# Feature toggles
audio_mode: str = Input(choices=["off", "on"])
reference_image: str = Input(description="Optional reference image URL")
```

### 3. Pipeline Methods

- `_auto_select_parameters()`: AI-powered parameter selection
- `_enhance_prompt()`: Professional prompt enhancement
- `_generate_image()`: Image generation with Ideogram v3
- `_generate_image_with_reference()`: Style transfer with Flux Kontext Pro
- `_generate_video()`: Video animation with SeDance-1-Pro
- `_generate_audio_for_video()`: Audio addition with MMAudio v2

---

## API Integration Guide

### OpenAI Integration

```python
# Initialize client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Structured output example
response = openai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format=YourPydanticModel,
    temperature=0.7
)

# Vision API example
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image..."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
)
```

### Replicate Integration

```python
# Set environment variable
os.environ["REPLICATE_API_TOKEN"] = replicate_token

# Image generation
output = replicate.run(
    "ideogram-ai/ideogram-v3-quality",
    input={
        "prompt": enhanced_prompt,
        "aspect_ratio": "16:9",
        "magic_prompt_option": "On"
    }
)

# Video generation
output = replicate.run(
    "bytedance/seedance-1-pro",
    input={
        "image": image_url,
        "prompt": video_prompt,
        "duration": 5,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "fps": 24
    }
)
```

### Langfuse Integration

```python
# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

# Decorator for tracking
@observe(name="function_name")
def your_function():
    # OpenAI calls are automatically tracked
    pass
```

---

## Adding Input Fields

### Step 1: Define Input Parameter

```python
# In predict() method signature
new_parameter: str = Input(
    description="Description of your new parameter",
    choices=["option1", "option2", "option3"],  # For dropdown
    default="option1"
)

# For different input types:
# String input
text_input: str = Input(description="Enter text")

# Integer input
number_input: int = Input(description="Enter number", ge=1, le=100)

# Float input
float_input: float = Input(description="Enter decimal", ge=0.0, le=1.0)

# Boolean input
boolean_input: bool = Input(description="Enable feature", default=False)

# File input
file_input: Path = Input(description="Upload file")
```

### Step 2: Process Input in Pipeline

```python
def predict(self, prompt: str, new_parameter: str, ...):
    # Access your parameter
    print(f"New parameter value: {new_parameter}")

    # Use in pipeline logic
    if new_parameter == "option1":
        # Custom logic
        pass

    # Pass to helper methods
    result = self._custom_method(prompt, new_parameter)
```

### Step 3: Update Helper Methods

```python
def _custom_method(self, prompt: str, new_parameter: str):
    # Use the new parameter in your custom logic
    if new_parameter == "special_mode":
        # Special processing
        return self._special_processing(prompt)
    else:
        # Default processing
        return self._default_processing(prompt)
```

---

## Writing Custom Pipelines

### 1. Create Pipeline Method

```python
def _custom_pipeline_step(self, input_data: str, parameters: dict) -> str:
    """
    Custom pipeline step template

    Args:
        input_data: Data from previous step
        parameters: Configuration parameters

    Returns:
        Processed data for next step
    """
    try:
        print("[custom] Starting custom pipeline step...")

        # Your custom logic here
        processed_data = self._process_data(input_data, parameters)

        print("[custom] Custom step completed successfully")
        return processed_data

    except Exception as e:
        print(f"[custom] Custom step failed: {e}")
        # Fallback logic
        return self._fallback_processing(input_data)
```

### 2. Integrate into Main Pipeline

```python
def predict(self, prompt: str, ...):
    try:
        # Existing steps...
        enhanced_prompt = self._enhance_prompt(prompt, openai_client)

        # Add your custom step
        custom_result = self._custom_pipeline_step(enhanced_prompt, selected_params)

        # Continue with remaining steps...
        image_url = self._generate_image(custom_result)

    except Exception as e:
        raise Exception(f"Pipeline failed: {str(e)}")
```

### 3. Error Handling & Fallbacks

```python
def _robust_pipeline_step(self, input_data: str) -> str:
    """Pipeline step with proper error handling"""

    try:
        # Primary processing
        return self._primary_processing(input_data)

    except SpecificException as e:
        print(f"[pipeline] Primary method failed: {e}, trying fallback")
        # Specific fallback
        return self._fallback_method(input_data)

    except Exception as e:
        print(f"[pipeline] All methods failed: {e}, using default")
        # Default fallback
        return self._default_result(input_data)
```

---

## Model Integration

### Adding New AI Models

#### 1. Research Model Requirements

```python
# Check model documentation for:
# - Input parameters
# - Output format
# - API endpoints
# - Rate limits
# - Pricing
```

#### 2. Create Model Wrapper

```python
def _generate_with_new_model(self, prompt: str, parameters: dict) -> str:
    """Wrapper for new AI model"""

    try:
        print("[new_model] Calling new AI model...")

        # Prepare model input
        model_input = {
            "prompt": prompt,
            "parameter1": parameters.get("param1", "default"),
            "parameter2": parameters.get("param2", 1.0)
        }

        # Call model
        output = replicate.run(
            "provider/model-name:version",
            input=model_input
        )

        print("[new_model] Generation completed successfully")
        return str(output)

    except Exception as e:
        print(f"[new_model] Model failed: {e}")
        raise Exception(f"New model generation failed: {str(e)}")
```

#### 3. Add Model Selection Logic

```python
def _generate_with_model_selection(self, prompt: str, model_choice: str) -> str:
    """Select and use appropriate model"""

    if model_choice == "ideogram":
        return self._generate_image(prompt)
    elif model_choice == "flux":
        return self._generate_image_with_reference(prompt, reference)
    elif model_choice == "new_model":
        return self._generate_with_new_model(prompt, parameters)
    else:
        raise ValueError(f"Unknown model: {model_choice}")
```

### Structured Output with Pydantic

#### 1. Define Response Model

```python
from pydantic import BaseModel
from typing import Literal, List

class CustomModelResponse(BaseModel):
    result: str
    confidence: float
    alternatives: List[str]
    reasoning: str
```

#### 2. Use with OpenAI

```python
def _structured_ai_call(self, prompt: str) -> CustomModelResponse:
    """Make structured AI call"""

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Your system prompt"},
            {"role": "user", "content": prompt}
        ],
        response_format=CustomModelResponse,
        temperature=0.7
    )

    result = response.choices[0].message.parsed
    if result is None:
        raise Exception("Failed to parse structured response")

    return result
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. API Key Errors

```python
# Check environment variables
def _validate_api_keys(self):
    required_keys = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        raise ValueError(f"Missing API keys: {missing_keys}")
```

#### 2. Model Timeout Issues

```python
# Add retry logic
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def _reliable_model_call(self, input_data):
    return replicate.run("model", input=input_data)
```

#### 3. Memory Management

```python
import gc

def _cleanup_resources(self):
    """Clean up resources after processing"""
    # Clear large variables
    gc.collect()

    # Remove temporary files
    temp_files = glob.glob("/tmp/outputs/temp_*")
    for file in temp_files:
        try:
            os.remove(file)
        except:
            pass
```

### Debug Mode

```python
def predict(self, prompt: str, debug_mode: bool = False, ...):
    if debug_mode:
        print(f"[DEBUG] Input prompt: {prompt}")
        print(f"[DEBUG] Selected parameters: {selected_params}")
        # Add more debug information

    # Continue with pipeline...
```

---

## Advanced Customization

### 1. Custom Prompt Templates

```python
CUSTOM_PROMPT_TEMPLATES = {
    "product_photography": """
    Create a professional product photography description for {product}.
    Style: {style}
    Lighting: {lighting}
    Background: {background}
    """,

    "lifestyle_banner": """
    Design a lifestyle banner featuring {product} in {environment}.
    Mood: {mood}
    Target audience: {audience}
    """
}

def _apply_custom_template(self, template_name: str, **kwargs) -> str:
    template = CUSTOM_PROMPT_TEMPLATES.get(template_name)
    if not template:
        raise ValueError(f"Unknown template: {template_name}")

    return template.format(**kwargs)
```

### 2. Dynamic Parameter Selection

```python
def _dynamic_parameter_selection(self, prompt: str, context: dict) -> dict:
    """Dynamically select parameters based on context"""

    # Analyze prompt content
    if "coffee" in prompt.lower():
        return {
            "video_style": "product",
            "lighting": "natural",
            "camera_movement": "push_in"
        }
    elif "tech" in prompt.lower():
        return {
            "video_style": "cinematic",
            "lighting": "studio",
            "camera_movement": "move_right"
        }
    # Add more rules...
```

### 3. Pipeline Composition

```python
class PipelineBuilder:
    def __init__(self):
        self.steps = []

    def add_step(self, step_function, **kwargs):
        self.steps.append((step_function, kwargs))
        return self

    def execute(self, initial_input):
        current_input = initial_input

        for step_function, kwargs in self.steps:
            current_input = step_function(current_input, **kwargs)

        return current_input

# Usage
def _custom_pipeline(self, prompt: str) -> str:
    pipeline = PipelineBuilder()
    pipeline.add_step(self._enhance_prompt, client=openai_client)
    pipeline.add_step(self._generate_image, style="product")
    pipeline.add_step(self._generate_video, duration=5)

    return pipeline.execute(prompt)
```

---

## Adding New Tools & Services

### Overview

The Banner Animator system is designed to be modular and extensible. You can add new AI models, processing tools, or external services by following these structured steps.

### Types of Tools You Can Add

#### 1. **AI Models**

- Image generation models (DALL-E, Midjourney, etc.)
- Video processing models (RunwayML, Pika Labs, etc.)
- Audio generation models (ElevenLabs, Mubert, etc.)
- Text processing models (Claude, Gemini, etc.)

#### 2. **Processing Tools**

- Image editors (PhotoRoom, Remove.bg, etc.)
- Video editors (FFmpeg wrappers, etc.)
- Audio processors (noise reduction, enhancement, etc.)
- Format converters

#### 3. **External Services**

- Cloud storage (AWS S3, Google Cloud, etc.)
- CDN services (Cloudflare, etc.)
- Analytics platforms (Mixpanel, etc.)
- Notification services (SendGrid, etc.)

---

### Step-by-Step Tool Integration Process

#### **Step 1: Research & Planning**

```python
# Document your new tool
NEW_TOOL_SPEC = {
    "name": "tool_name",
    "type": "ai_model",  # or "processing_tool", "external_service"
    "purpose": "What this tool does",
    "input_format": "Expected input format",
    "output_format": "Expected output format",
    "api_endpoint": "API URL or library import",
    "rate_limits": "API limitations",
    "pricing": "Cost considerations",
    "dependencies": ["required", "packages"]
}
```

#### **Step 2: Add Dependencies**

```yaml
# Update cog.yaml
build:
  python_packages:
    - "existing-packages"
    - "new-tool-sdk"
    - "new-tool-dependencies"
```

```bash
# Install locally for testing
pip install new-tool-sdk
```

#### **Step 3: Environment Configuration**

```env
# Add to .env file
NEW_TOOL_API_KEY=your_api_key_here
NEW_TOOL_BASE_URL=https://api.newtool.com
NEW_TOOL_MODEL_VERSION=v1.0
```

#### **Step 4: Create Tool Wrapper Class**

```python
class NewToolWrapper:
    """Wrapper for integrating new AI tool"""

    def __init__(self):
        self.api_key = os.getenv("NEW_TOOL_API_KEY")
        self.base_url = os.getenv("NEW_TOOL_BASE_URL")
        self.model_version = os.getenv("NEW_TOOL_MODEL_VERSION", "v1.0")

        if not self.api_key:
            raise ValueError("NEW_TOOL_API_KEY not found in environment")

    def process(self, input_data: str, parameters: dict = None) -> str:
        """Main processing method"""
        try:
            # Prepare request
            request_data = self._prepare_request(input_data, parameters)

            # Make API call
            response = self._make_api_call(request_data)

            # Process response
            result = self._process_response(response)

            return result

        except Exception as e:
            raise Exception(f"NewTool processing failed: {str(e)}")

    def _prepare_request(self, input_data: str, parameters: dict) -> dict:
        """Prepare API request payload"""
        return {
            "input": input_data,
            "model": self.model_version,
            "parameters": parameters or {}
        }

    def _make_api_call(self, request_data: dict) -> dict:
        """Make the actual API call"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/process",
            json=request_data,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()
        return response.json()

    def _process_response(self, response: dict) -> str:
        """Extract and format the result"""
        if "result" in response:
            return response["result"]
        elif "output" in response:
            return response["output"]
        else:
            raise Exception("Unexpected response format")
```

#### **Step 5: Add Tool Input Parameter**

```python
def predict(
    self,
    prompt: str = Input(description="Describe what you want to create"),
    # ... existing parameters ...
    use_new_tool: bool = Input(
        description="Enable new tool processing",
        default=False
    ),
    new_tool_settings: str = Input(
        description="Settings for new tool",
        choices=["standard", "enhanced", "creative"],
        default="standard"
    ),
):
```

#### **Step 6: Integrate into Pipeline**

```python
def predict(self, prompt: str, use_new_tool: bool, new_tool_settings: str, ...):
    try:
        # ... existing pipeline steps ...

        # Add new tool step
        if use_new_tool:
            print("[new_tool] Processing with new tool...")
            processed_result = self._process_with_new_tool(
                enhanced_prompt,
                new_tool_settings
            )
            # Use processed_result in subsequent steps
        else:
            processed_result = enhanced_prompt

        # Continue with existing pipeline...

    except Exception as e:
        raise Exception(f"Pipeline failed: {str(e)}")

@observe(name="new_tool_processing")
def _process_with_new_tool(self, input_data: str, settings: str) -> str:
    """Process data using the new tool"""
    try:
        # Initialize tool
        tool = NewToolWrapper()

        # Prepare parameters based on settings
        parameters = self._get_tool_parameters(settings)

        # Process data
        result = tool.process(input_data, parameters)

        print(f"[new_tool] Processing completed successfully")
        return result

    except Exception as e:
        print(f"[new_tool] Processing failed: {e}, using fallback")
        # Fallback to original data
        return input_data

def _get_tool_parameters(self, settings: str) -> dict:
    """Get parameters based on user settings"""
    parameter_map = {
        "standard": {"quality": "normal", "speed": "fast"},
        "enhanced": {"quality": "high", "speed": "medium"},
        "creative": {"quality": "high", "creativity": "max", "speed": "slow"}
    }
    return parameter_map.get(settings, parameter_map["standard"])
```

#### **Step 7: Add Error Handling & Fallbacks**

```python
def _robust_tool_integration(self, input_data: str, tool_config: dict) -> str:
    """Robust tool integration with multiple fallback strategies"""

    # Try primary tool
    try:
        return self._process_with_new_tool(input_data, tool_config)
    except Exception as e:
        print(f"[tool] Primary tool failed: {e}")

    # Try alternative tool
    try:
        return self._process_with_alternative_tool(input_data, tool_config)
    except Exception as e:
        print(f"[tool] Alternative tool failed: {e}")

    # Try local processing
    try:
        return self._process_locally(input_data, tool_config)
    except Exception as e:
        print(f"[tool] Local processing failed: {e}")

    # Final fallback - return original data
    print(f"[tool] All processing methods failed, using original data")
    return input_data
```

#### **Step 8: Add Validation & Testing**

```python
def _validate_new_tool(self) -> bool:
    """Validate that new tool is properly configured"""
    try:
        tool = NewToolWrapper()

        # Test with minimal input
        test_result = tool.process("test input", {"test": True})

        # Validate result format
        if isinstance(test_result, str) and len(test_result) > 0:
            print("[validation] New tool validation successful")
            return True
        else:
            print("[validation] New tool returned invalid result")
            return False

    except Exception as e:
        print(f"[validation] New tool validation failed: {e}")
        return False

def setup(self):
    """Enhanced setup with tool validation"""
    # ... existing setup ...

    # Validate new tools
    if os.getenv("NEW_TOOL_API_KEY"):
        if not self._validate_new_tool():
            print("[setup] Warning: New tool validation failed")
```

#### **Step 9: Add Configuration Options**

```python
# Add to PROMPT_STRUCTURE or create new config
NEW_TOOL_CONFIG = {
    "processing_modes": {
        "fast": "Quick processing with standard quality",
        "balanced": "Balanced speed and quality",
        "quality": "High quality processing, slower speed",
        "creative": "Maximum creativity, longest processing time"
    },
    "output_formats": {
        "standard": "Standard output format",
        "enhanced": "Enhanced output with metadata",
        "detailed": "Detailed output with analysis"
    }
}

def _configure_new_tool(self, user_preferences: dict) -> dict:
    """Configure new tool based on user preferences"""
    config = {
        "mode": user_preferences.get("processing_mode", "balanced"),
        "format": user_preferences.get("output_format", "standard"),
        "quality": user_preferences.get("quality_level", "medium")
    }
    return config
```

#### **Step 10: Update Documentation**

```python
# Add to system prompt or create new documentation
NEW_TOOL_DOCUMENTATION = """
New Tool Integration:
- Purpose: {tool_purpose}
- Input: {input_format}
- Output: {output_format}
- Parameters: {available_parameters}
- Use Cases: {common_use_cases}
"""

def _get_tool_usage_help(self) -> str:
    """Return help information for new tool"""
    return NEW_TOOL_DOCUMENTATION.format(
        tool_purpose="Enhances image quality and adds artistic effects",
        input_format="Text prompts or image URLs",
        output_format="Enhanced image URLs or processed text",
        available_parameters="quality, style, creativity, speed",
        common_use_cases="Image enhancement, style transfer, quality upscaling"
    )
```

---

### **Advanced Tool Integration Patterns**

#### **1. Conditional Tool Usage**

```python
def _smart_tool_selection(self, input_data: str, context: dict) -> str:
    """Intelligently select which tools to use based on context"""

    # Analyze input to determine best tools
    if "image" in context.get("input_type", ""):
        if context.get("quality") == "low":
            return self._process_with_upscaler(input_data)
        elif context.get("style") == "artistic":
            return self._process_with_style_transfer(input_data)

    elif "audio" in context.get("input_type", ""):
        if context.get("has_noise", False):
            return self._process_with_noise_reduction(input_data)

    # Default processing
    return self._process_with_standard_tools(input_data)
```

#### **2. Tool Chaining**

```python
def _chain_tools(self, input_data: str, tool_chain: List[str]) -> str:
    """Chain multiple tools in sequence"""
    current_data = input_data

    for tool_name in tool_chain:
        print(f"[chain] Processing with {tool_name}...")
        current_data = self._process_with_tool(current_data, tool_name)

    return current_data

def _process_with_tool(self, data: str, tool_name: str) -> str:
    """Process data with specified tool"""
    tool_map = {
        "enhancer": self._process_with_enhancer,
        "upscaler": self._process_with_upscaler,
        "style_transfer": self._process_with_style_transfer,
        "new_tool": self._process_with_new_tool
    }

    processor = tool_map.get(tool_name)
    if not processor:
        raise ValueError(f"Unknown tool: {tool_name}")

    return processor(data)
```

#### **3. Parallel Tool Processing**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def _parallel_tool_processing(self, input_data: str, tools: List[str]) -> dict:
    """Process input with multiple tools in parallel"""

    def process_with_single_tool(tool_name: str) -> tuple:
        try:
            result = self._process_with_tool(input_data, tool_name)
            return tool_name, result, None
        except Exception as e:
            return tool_name, None, str(e)

    # Process in parallel
    with ThreadPoolExecutor(max_workers=len(tools)) as executor:
        futures = [executor.submit(process_with_single_tool, tool) for tool in tools]
        results = {}

        for future in futures:
            tool_name, result, error = future.result()
            results[tool_name] = {
                "result": result,
                "error": error,
                "success": error is None
            }

    return results
```

---

### **Tool Integration Best Practices**

#### **1. Configuration Management**

```python
class ToolConfig:
    """Centralized tool configuration"""

    def __init__(self):
        self.tools = {}
        self.load_configurations()

    def load_configurations(self):
        """Load tool configurations from environment or config files"""
        self.tools = {
            "new_tool": {
                "enabled": os.getenv("NEW_TOOL_ENABLED", "false").lower() == "true",
                "api_key": os.getenv("NEW_TOOL_API_KEY"),
                "base_url": os.getenv("NEW_TOOL_BASE_URL"),
                "timeout": int(os.getenv("NEW_TOOL_TIMEOUT", "60")),
                "retry_attempts": int(os.getenv("NEW_TOOL_RETRY", "3"))
            }
        }

    def get_tool_config(self, tool_name: str) -> dict:
        """Get configuration for specific tool"""
        return self.tools.get(tool_name, {})

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if tool is enabled"""
        return self.get_tool_config(tool_name).get("enabled", False)
```

#### **2. Usage Monitoring**

```python
def _track_tool_usage(self, tool_name: str, input_size: int, duration: float, success: bool):
    """Track tool usage for monitoring and optimization"""

    usage_data = {
        "tool": tool_name,
        "timestamp": datetime.now().isoformat(),
        "input_size": input_size,
        "duration": duration,
        "success": success
    }

    # Log to monitoring service
    if langfuse:
        langfuse.track(event="tool_usage", properties=usage_data)

    # Log locally
    print(f"[monitoring] {tool_name}: {duration:.2f}s, success: {success}")
```

#### **3. Cost Management**

```python
def _estimate_tool_cost(self, tool_name: str, input_data: str) -> float:
    """Estimate cost for using specific tool"""

    cost_map = {
        "new_tool": {
            "base_cost": 0.01,  # Base cost per request
            "per_char": 0.0001,  # Cost per character
            "per_minute": 0.05   # For video/audio processing
        }
    }

    tool_pricing = cost_map.get(tool_name, {})

    estimated_cost = tool_pricing.get("base_cost", 0)
    estimated_cost += len(input_data) * tool_pricing.get("per_char", 0)

    return estimated_cost

def _check_budget_limits(self, estimated_cost: float) -> bool:
    """Check if tool usage is within budget limits"""
    budget_limit = float(os.getenv("DAILY_BUDGET_LIMIT", "100.0"))
    current_usage = self._get_daily_usage()

    return (current_usage + estimated_cost) <= budget_limit
```

---

This comprehensive guide provides everything developers need to successfully integrate new tools into the Banner Animator system. The step-by-step process ensures proper integration while maintaining system reliability and performance.

---

## Best Practices

### 1. Code Organization

- Keep each pipeline step as a separate method
- Use consistent naming conventions (`_method_name`)
- Add comprehensive error handling
- Include logging for debugging

### 2. Performance Optimization

- Use parallel API calls when possible
- Implement caching for repeated operations
- Clean up temporary files
- Monitor memory usage

### 3. Error Handling

- Always provide fallback options
- Log errors with context
- Use structured error messages
- Implement retry mechanisms

### 4. Testing

```python
def test_pipeline_step(self):
    """Test individual pipeline steps"""
    test_input = "test prompt"
    result = self._enhance_prompt(test_input, mock_client)
    assert result is not None
    assert len(result) > len(test_input)
```

---

## Deployment

### 1. Cog Deployment

```bash
# Build the model
cog build -t banner-animator

# Push to registry
cog push r8.im/your-username/banner-animator

# Deploy on Replicate
# Visit replicate.com and create new model
```

### 2. Environment Setup

```bash
# Production environment variables
export OPENAI_API_KEY="your-production-key"
export REPLICATE_API_TOKEN="your-production-token"
# ... other variables
```

### 3. Monitoring

- Set up Langfuse for tracking
- Monitor API usage and costs
- Set up alerts for failures
- Track performance metrics

---

This documentation provides a complete guide for understanding, replicating, and extending the Banner Animator Generator system. Each section includes practical examples and can be used as a reference for development and maintenance.
