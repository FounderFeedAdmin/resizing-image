# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import requests
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath, Secret
from openai import OpenAI
import replicate
from pydantic import BaseModel
import base64


class ImageResizePrompt(BaseModel):
    """Structured response for image resize analysis"""

    resize_prompt: str
    aspect_ratio_description: str
    layout_adjustments: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("[setup] Initializing Image Resizing Tool with Seedream-4...")

        # Create output directories for saving results
        os.makedirs("/tmp/outputs", exist_ok=True)

        print("[setup] Image Resizing Tool with Seedream-4 initialized successfully")

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
        openai_api_key: Secret = Input(description="OpenAI API key for image analysis"),
        replicate_api_token: Secret = Input(
            description="Replicate API token for image generation"
        ),
        debug_mode: bool = Input(
            description="Enable debug mode",
            default=False,
        ),
    ) -> CogPath:
        """Resize image to target aspect ratio using Seedream-4 while maintaining content integrity"""
        try:
            print(f"[resize] Starting image resize process...")
            print(f"[resize] Target aspect ratio: {aspect_ratio}")
            print(f"[resize] Quality level: {quality}")

            # Initialize OpenAI client with secret input
            openai_client = OpenAI(api_key=openai_api_key.get_secret_value())

            # Set Replicate API token from secret input
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_token.get_secret_value()

            self._validate_resize_inputs(input_image, aspect_ratio)

            # Step 1: Prepare image input (URL or base64)
            if self._is_url(input_image):
                image_input = {"url": input_image}
                print("[input] Using public URL for image input.")
            else:
                image_input = {"base64": self._file_to_base64(input_image)}
                print("[input] Using base64-encoded image input.")

            # Step 2: Analyze image with OpenAI to generate a detailed resize prompt
            resize_analysis = self._analyze_image_for_resize(
                openai_client, image_input, aspect_ratio
            )

            # Step 3: Generate resized image using Replicate (Seedream-4)
            resized_image_url = self._resize_with_seedream4(
                image_input, resize_analysis.resize_prompt, aspect_ratio, quality
            )

            # Step 4: Download and return the final image
            final_image_path = self._download_final_image(resized_image_url)

            print(f"[resize] Image resizing completed successfully")
            return final_image_path

        except Exception as e:
            print(f"[resize] Error in resize pipeline: {str(e)}")
            return input_image

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

    def _analyze_image_for_resize(
        self, openai_client: OpenAI, image_input: dict, target_aspect_ratio: str
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

            response = openai_client.beta.chat.completions.parse(
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

    def _resize_with_seedream4(
        self,
        image_input: dict,
        resize_prompt: str,
        aspect_ratio: str,
        quality: str,
    ) -> str:
        """
        Generate resized image using Seedream-4.
        Accepts either a public URL or base64-encoded image.
        """
        try:
            print(f"[seedream4] Generating resized image with Seedream-4...")

            # Map quality levels to image sizes
            quality_settings = {
                "standard": "1K",  # 1024px
                "high": "2K",  # 2048px
                "ultra": "4K",  # 4096px
            }
            size = quality_settings.get(quality, quality_settings["high"])

            # Prepare input for Replicate
            seedream_input = {
                "prompt": resize_prompt,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "sequential_image_generation": "disabled",
                "max_images": 1,
            }

            # Add image input as either URL or base64
            if "url" in image_input:
                seedream_input["image_input"] = [image_input["url"]]
            elif "base64" in image_input:
                seedream_input["image_input"] = [
                    f"data:image/png;base64,{image_input['base64']}"
                ]

            print(f"[seedream4] Calling Seedream-4 with {size} resolution...")
            output = replicate.run("bytedance/seedream-4", input=seedream_input)

            # Seedream-4 returns an array of URLs, we take the first one
            if isinstance(output, list) and len(output) > 0:
                result_url = output[0]
            else:
                result_url = str(output)

            print(f"[seedream4] Image generation completed successfully")
            return result_url

        except Exception as e:
            print(f"[seedream4] Seedream-4 generation failed: {e}")
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
