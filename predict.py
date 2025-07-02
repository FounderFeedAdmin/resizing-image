# Prediction interface for Cog ⚙️
# https://cog.run/python

from dotenv import load_dotenv

load_dotenv()  # Always load .env at the start

import os
import requests
import tempfile
from pathlib import Path
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath
from openai import OpenAI
import replicate
from pydantic import BaseModel
import base64

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
except ImportError:
    # Fallback if Langfuse is not installed
    Langfuse = None

    def observe(name=None):
        def decorator(func):
            return func

        return decorator


import cloudinary
import cloudinary.uploader


class ImageResizePrompt(BaseModel):
    """Structured response for image resize analysis"""

    resize_prompt: str
    aspect_ratio_description: str
    layout_adjustments: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("[setup] Initializing Image Resizing Tool...")

        # Create output directories for saving results
        os.makedirs("/tmp/outputs", exist_ok=True)

        # Initialize OpenAI client (optional during build, required during prediction)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
            print("[setup] OpenAI client initialized")
        else:
            self.openai_client = None
            print(
                "[setup] OpenAI API key not found - will initialize during prediction"
            )

        # Set Replicate API token for model calls
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if replicate_token:
            os.environ["REPLICATE_API_TOKEN"] = replicate_token

        # Configure Cloudinary for image uploads
        if all(
            [
                os.getenv("NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME"),
                os.getenv("CLOUDINARY_API_KEY"),
                os.getenv("CLOUDINARY_API_SECRET"),
            ]
        ):
            cloudinary.config(
                cloud_name=os.getenv("NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME"),
                api_key=os.getenv("CLOUDINARY_API_KEY"),
                api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            )

        # Initialize Langfuse for tracking (optional, does not break if missing)
        self.langfuse = None
        if (
            Langfuse is not None
            and os.getenv("LANGFUSE_SECRET_KEY")
            and os.getenv("LANGFUSE_PUBLIC_KEY")
        ):
            try:
                self.langfuse = Langfuse(
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    host="https://cloud.langfuse.com",
                )
                print("[setup] Langfuse initialized for tracking")
            except Exception as e:
                print(f"[setup] Langfuse initialization failed: {e}")
                self.langfuse = None

        print("[setup] Image Resizing Tool initialized successfully")

    def _is_url(self, input_image) -> bool:
        """Check if the input is a public URL."""
        return isinstance(input_image, str) and input_image.startswith("http")

    def _file_to_base64(self, file_path: str | Path) -> str:
        """Convert a local image file to a base64-encoded string."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def predict(
        self,
        input_image: CogPath = Input(description="Input image to resize (URL or file)"),
        aspect_ratio: str = Input(
            description="Target aspect ratio for the resized image",
            choices=["16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "21:9", "9:21"],
            default="16:9",
        ),
        quality: str = Input(
            description="Quality level for the resized image",
            choices=["standard", "high", "ultra"],
            default="high",
        ),
    ) -> CogPath:
        """Resize image to target aspect ratio while maintaining content integrity"""
        try:
            print(f"[resize] Starting image resize process...")
            print(f"[resize] Target aspect ratio: {aspect_ratio}")
            print(f"[resize] Quality level: {quality}")

            self._ensure_openai_client()
            self._validate_resize_inputs(input_image, aspect_ratio)

            # Step 1: Prepare image input (URL or base64)
            if self._is_url(input_image):
                image_input = {"url": input_image}
                print("[input] Using public URL for image input.")
            else:
                image_input = {"base64": self._file_to_base64(input_image)}
                print("[input] Using base64-encoded image input.")

            # Step 2: Analyze image with OpenAI to generate a detailed resize prompt
            resize_analysis = self._analyze_image_for_resize(image_input, aspect_ratio)

            # Step 3: Generate resized image using Replicate (Flux Kontext Dev)
            resized_image_url = self._resize_with_flux_kontext(
                image_input, resize_analysis.resize_prompt, aspect_ratio, quality
            )

            # Step 4: Download and return the final image
            final_image_path = self._download_final_image(resized_image_url)

            print(f"[resize] Image resizing completed successfully")
            return final_image_path

        except Exception as e:
            print(f"[resize] Error in resize pipeline: {str(e)}")
            return input_image

    def _ensure_openai_client(self):
        """Ensure OpenAI client is initialized (for runtime prediction)"""
        if self.openai_client is None:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for predictions"
                )
            self.openai_client = OpenAI(api_key=openai_key)
            print("[client] OpenAI client initialized for prediction")

    def _validate_resize_inputs(self, input_image: CogPath, aspect_ratio: str) -> None:
        """Validate input parameters: image must exist, aspect ratio must be valid"""
        if not input_image or not os.path.exists(input_image):
            raise ValueError("Input image is required and must exist")

        valid_ratios = [
            "16:9",
            "9:16",
            "1:1",
            "4:3",
            "3:4",
            "4:5",
            "5:4",
            "21:9",
            "9:21",
        ]
        if aspect_ratio not in valid_ratios:
            raise ValueError(f"Invalid aspect ratio. Must be one of: {valid_ratios}")

        print(f"[validation] Input validation passed")

    def _upload_image_to_cloudinary(self, image_path: CogPath) -> str:
        """Upload image to Cloudinary and return a public URL for downstream use"""
        try:
            print("[upload] Uploading image to Cloudinary...")

            result = cloudinary.uploader.upload(
                str(image_path), folder="resizing_tool", resource_type="image"
            )

            image_url = result["secure_url"]
            print(f"[upload] Image uploaded successfully")
            return image_url

        except Exception as e:
            print(f"[upload] Cloudinary upload failed: {e}")
            # Fallback: convert to base64 or use local path (not implemented here)
            raise Exception(f"Failed to upload image: {str(e)}")

    @observe(name="analyze_image_for_resize")
    def _analyze_image_for_resize(
        self, image_input: dict, target_aspect_ratio: str
    ) -> ImageResizePrompt:
        """
        Analyze image with OpenAI and generate a detailed resize prompt.
        Accepts either a public URL or base64-encoded image.
        """
        try:
            print(f"[analysis] Analyzing image for {target_aspect_ratio} resize...")
            system_prompt = """You are a professional commercial photographer and marketing specialist who analyzes existing marketing banners and creates detailed photo-realistic descriptions for resizing them to new aspect ratios.

**CORE TASK**: Analyze the provided image and create a detailed prompt that recreates the same marketing banner in the target aspect ratio while preserving all visual elements.

**ANALYSIS FRAMEWORK**:
- **Visual Elements**: All text, logos, products, decorative items, backgrounds
- **Color Palette**: Exact colors, gradients, and color relationships  
- **Layout Composition**: How elements are positioned and balanced
- **Lighting & Effects**: Shadows, highlights, atmospheric effects
- **Design Style**: Modern, retro, minimalist, etc.
- **Text Placement**: Where text should be repositioned for the new aspect ratio

**RESIZE STRATEGY**:
1. **Never crop or distort**: Extend canvas to fit new aspect ratio
2. **Preserve all elements**: Every logo, text, product, decoration must remain
3. **Smart repositioning**: Optimize layout for new dimensions
4. **Background extension**: Seamlessly extend backgrounds to fill new space
5. **Maintain hierarchy**: Keep important elements prominent

**OUTPUT FORMAT**: 
Generate a detailed photo-realistic description that starts with "A professional marketing banner in [aspect_ratio] format featuring..." and includes:
- All visible text exactly as shown
- Product descriptions and positioning
- Background details and extensions
- Color schemes and lighting
- Overall composition and style

**EXAMPLE OUTPUT**:
"A professional marketing banner in 16:9 format featuring the text 'SUMMER SALE 50% OFF' in bold white letters on the left side. A tropical orange and teal gradient background with palm leaf silhouettes extends across the entire width. The red circular badge with '50% OFF' is positioned in the bottom right corner. Additional decorative palm leaves frame the composition with natural green tones. The layout maintains the vibrant summer aesthetic with extended background gradients filling the wider format while keeping all text and graphics clearly visible and properly balanced."

Respond with the resize prompt, aspect ratio description, and layout adjustments in structured format."""

            user_prompt = f"Analyze this image and create a resize prompt for {target_aspect_ratio} aspect ratio. Maintain all visual elements while optimizing the layout for the new dimensions."

            if self.openai_client is None:
                raise ValueError("OpenAI client not initialized")

            # Prepare OpenAI message content
            content = [
                {"type": "text", "text": user_prompt},
            ]
            if "url" in image_input:
                content.append(
                    {"type": "image_url", "image_url": {"url": image_input["url"]}}
                )
            elif "base64" in image_input:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_input['base64']}"
                        },
                    }
                )

            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                response_format=ImageResizePrompt,
                temperature=0.7,
                max_tokens=1000,
            )

            result = response.choices[0].message.parsed
            if result is None:
                raise Exception("Failed to parse OpenAI response")

            print(f"[analysis] Image analysis completed successfully")
            return result

        except Exception as e:
            print(f"[analysis] OpenAI analysis failed: {e}")
            fallback_prompt = f"Professional marketing banner resized to {target_aspect_ratio} aspect ratio, maintaining all original visual elements, text, and branding with optimized layout and extended background"
            return ImageResizePrompt(
                resize_prompt=fallback_prompt,
                aspect_ratio_description=f"Resized to {target_aspect_ratio}",
                layout_adjustments="Optimized layout for new aspect ratio",
            )

    @observe(name="resize_with_flux_kontext")
    def _resize_with_flux_kontext(
        self,
        image_input: dict,
        resize_prompt: str,
        aspect_ratio: str,
        quality: str,
    ) -> str:
        """
        Generate resized image using Flux Kontext Dev.
        Accepts either a public URL or base64-encoded image.
        """
        try:
            print(f"[flux] Generating resized image with Flux Kontext Dev...")
            quality_settings = {
                "standard": {"num_inference_steps": 20, "guidance": 2.0},
                "high": {"num_inference_steps": 30, "guidance": 2.5},
                "ultra": {"num_inference_steps": 40, "guidance": 3.0},
            }
            settings = quality_settings.get(quality, quality_settings["high"])

            # Prepare input for Replicate
            flux_input = {
                "prompt": resize_prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",
                "num_inference_steps": settings["num_inference_steps"],
                "guidance": settings["guidance"],
                "go_fast": False,
                "output_quality": 100,
                "disable_safety_checker": True,
            }
            # Add image input as either URL or base64
            if "url" in image_input:
                flux_input["input_image"] = image_input["url"]
            elif "base64" in image_input:
                flux_input["input_image"] = (
                    f"data:image/png;base64,{image_input['base64']}"
                )

            print(
                f"[flux] Calling Flux Kontext Dev with {settings['num_inference_steps']} steps..."
            )
            output = replicate.run(
                "black-forest-labs/flux-kontext-dev", input=flux_input
            )
            print(f"[flux] Image generation completed successfully")
            return str(output)

        except Exception as e:
            print(f"[flux] Flux Kontext Dev generation failed: {e}")
            raise Exception(f"Failed to generate resized image: {str(e)}")

    def _download_final_image(self, image_url: str) -> CogPath:
        """
        Download the final resized image from the given URL and save it to /tmp/outputs.
        Returns a CogPath for downstream use.
        """
        try:
            print(f"[download] Downloading final resized image...")

            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Create output file with a unique name based on image URL hash
            output_path = Path(
                f"/tmp/outputs/resized_image_{hash(image_url) % 100000}.jpg"
            )

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"[download] Image downloaded successfully to {output_path}")
            return CogPath(output_path)

        except Exception as e:
            print(f"[download] Failed to download image: {e}")
            raise Exception(f"Failed to download final image: {str(e)}")

    def _cleanup_temp_files(self) -> None:
        """
        Clean up temporary files in /tmp/outputs to avoid disk bloat.
        """
        try:
            import glob

            temp_files = glob.glob("/tmp/outputs/temp_*")
            for file in temp_files:
                try:
                    os.remove(file)
                except:
                    pass
        except:
            pass


# Helper to require environment variables


def require_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return value
