"""
Godot Image Generator
Simple bridge to OpenAI's gpt-image-1 for game asset generation.
Saves to a shared asset library for reuse across multiple projects.
Designed to be easily wrapped in an MCP server.
"""

import os
import base64
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Literal, List
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv


class AssetLibrary:
    """
    Manages the shared asset library and catalog.
    """
    
    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories
        self.categories = {
            'sprites': library_dir / 'sprites',
            'characters': library_dir / 'sprites' / 'characters',
            'enemies': library_dir / 'sprites' / 'enemies',
            'items': library_dir / 'sprites' / 'items',
            'tiles': library_dir / 'tiles',
            'ui': library_dir / 'ui',
            'backgrounds': library_dir / 'backgrounds',
            'effects': library_dir / 'effects'
        }
        
        for cat_path in self.categories.values():
            cat_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = library_dir / 'asset_index.json'
        self.cost_log_path = library_dir / 'cost_log.json'
        self.index = self._load_index()
        self.cost_data = self._load_cost_log()
    
    def _load_index(self) -> Dict:
        """Load the asset index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {'assets': [], 'total_assets': 0}
    
    def _save_index(self):
        """Save the asset index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _load_cost_log(self) -> Dict:
        """Load the cost log."""
        if self.cost_log_path.exists():
            with open(self.cost_log_path, 'r') as f:
                return json.load(f)
        return {'total_cost': 0.0, 'generations': []}
    
    def _save_cost_log(self):
        """Save the cost log."""
        with open(self.cost_log_path, 'w') as f:
            json.dump(self.cost_data, f, indent=2)
    
    def add_asset(
        self,
        file_path: Path,
        category: str,
        prompt: str,
        tags: List[str],
        metadata: Dict
    ):
        """Add an asset to the catalog."""
        asset_entry = {
            'id': hashlib.md5(str(file_path).encode()).hexdigest()[:12],
            'filename': file_path.name,
            'path': str(file_path.relative_to(self.library_dir)),
            'full_path': str(file_path),
            'category': category,
            'prompt': prompt,
            'tags': tags,
            'created': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        self.index['assets'].append(asset_entry)
        self.index['total_assets'] = len(self.index['assets'])
        self._save_index()
        
        return asset_entry
    
    def search_assets(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search for assets in the catalog."""
        results = self.index['assets']
        
        if category:
            results = [a for a in results if a['category'] == category]
        
        if tags:
            results = [
                a for a in results 
                if any(tag.lower() in [t.lower() for t in a.get('tags', [])] for tag in tags)
            ]
        
        if query:
            query_lower = query.lower()
            results = [
                a for a in results
                if query_lower in a['prompt'].lower() or
                   query_lower in a['filename'].lower() or
                   any(query_lower in tag.lower() for tag in a.get('tags', []))
            ]
        
        return results
    
    def log_generation(self, prompt: str, path: str, cost: float, metadata: Dict):
        """Log a generation to the cost log."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:100],
            'path': path,
            'cost': cost,
            **metadata
        }
        
        self.cost_data['generations'].append(log_entry)
        self.cost_data['total_cost'] += cost
        self._save_cost_log()
    
    def get_total_cost(self) -> float:
        """Get total cost of all generations."""
        return self.cost_data.get('total_cost', 0.0)
    
    def get_category_path(self, category: str) -> Path:
        """Get the path for a category."""
        return self.categories.get(category, self.library_dir)


class ImageGenerator:
    """
    Core image generation class.
    Clean interface designed to be easily wrapped by MCP tools.
    """
    
    # Quality to cost mapping (approximate, in USD)
    COST_MAP = {
        'low': 0.01,
        'medium': 0.04,
        'high': 0.17
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        library_dir: Optional[str] = None,
        enable_catalog: bool = True
    ):
        """
        Initialize the image generator.
        
        Args:
            api_key: OpenAI API key (if not provided, reads from env)
            library_dir: Path to shared asset library
            enable_catalog: Whether to catalog generated assets
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        library_path = Path(library_dir or os.getenv('ASSET_LIBRARY_DIR', './godot-assets'))
        self.library = AssetLibrary(library_path)
        self.enable_catalog = enable_catalog
        
        print(f"📁 Asset library: {library_path}")
        print(f"💰 Total spent: ${self.library.get_total_cost():.2f}")
    
    def generate(
        self,
        prompt: str,
        category: str = 'sprites',
        filename: Optional[str] = None,
        size: Literal['1024x1024', '1792x1024', '1024x1792'] = '1024x1024',
        quality: Literal['low', 'medium', 'high'] = 'medium',
        style: Optional[str] = None,
        transparent_background: bool = False,
        tags: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            category: Asset category (sprites, tiles, ui, backgrounds, etc.)
            filename: Output filename (auto-generated if not provided)
            size: Image dimensions
            quality: Generation quality (affects speed and cost)
            style: Optional style modifier (e.g., "pixel art", "hand-drawn")
            transparent_background: Whether to request transparent background
            tags: List of searchable tags for cataloging
            
        Returns:
            Dict containing asset information
        """
        # Enhance prompt with style and background if specified
        enhanced_prompt = prompt
        if style:
            enhanced_prompt = f"{style}, {enhanced_prompt}"
        if transparent_background:
            enhanced_prompt = f"{enhanced_prompt}, transparent background"
        
        print(f"🎨 Generating: {enhanced_prompt[:50]}...")
        
        try:
            # Call OpenAI API - gpt-image-1 only
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=enhanced_prompt,
                size=size,
                quality=quality
            )
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_prompt = safe_prompt.replace(' ', '_')
                filename = f"{safe_prompt}_{timestamp}.png"
            
            # Ensure .png extension
            if not filename.endswith('.png'):
                filename = f"{filename}.png"
            
            # Get category path and save
            category_path = self.library.get_category_path(category)
            output_path = category_path / filename
            
            # Download image from URL or decode base64
            import requests
            
            # gpt-image-1 returns both url and b64_json attributes
            # Check which one has actual data
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                # Use base64 data if available
                image_data = base64.b64decode(response.data[0].b64_json)
            elif hasattr(response.data[0], 'url') and response.data[0].url:
                # Fall back to downloading from URL
                image_url = response.data[0].url
                image_response = requests.get(image_url)
                image_data = image_response.content
            else:
                raise ValueError(f"Response has neither valid url nor b64_json: {response.data[0]}")
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            # Track cost
            cost = self.COST_MAP[quality]
            
            # Catalog the asset
            metadata = {
                'size': size,
                'quality': quality,
                'cost': cost,
                'transparent_background': transparent_background,
                'style': style
            }
            
            if self.enable_catalog:
                asset_entry = self.library.add_asset(
                    file_path=output_path,
                    category=category,
                    prompt=enhanced_prompt,
                    tags=tags or [],
                    metadata=metadata
                )
            else:
                asset_entry = {'path': str(output_path), 'filename': filename}
            
            self.library.log_generation(enhanced_prompt, str(output_path), cost, metadata)
            
            print(f"✓ Saved: {output_path.relative_to(self.library.library_dir)}")
            print(f"  Cost: ${cost:.2f} | Total: ${self.library.get_total_cost():.2f}")
            
            return {
                **asset_entry,
                'cost': cost,
                'category': category,
                'full_path': str(output_path)
            }
            
        except Exception as e:
            print(f"✗ Error generating image: {e}")
            raise
    
    def generate_sprite(
        self,
        description: str,
        subcategory: str = 'characters',
        size: str = "256x256",
        filename: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate a game sprite with optimized settings.
        
        Args:
            description: Description of the sprite
            subcategory: Sprite subcategory (characters, enemies, items)
            size: Sprite dimensions (e.g., "32x32", "64x64", "256x256")
            filename: Output filename
            tags: Searchable tags
            
        Returns:
            Asset information dict
        """
        openai_size = self._closest_openai_size(size)
        
        prompt = f"game sprite, {description}, simple, clean design, centered on canvas"
        
        # Determine category path
        category = f'sprites/{subcategory}' if subcategory in ['characters', 'enemies', 'items'] else 'sprites'
        
        # Auto-generate tags
        auto_tags = ['sprite', subcategory]
        if tags:
            auto_tags.extend(tags)
        
        return self.generate(
            prompt=prompt,
            category=category,
            filename=filename,
            size=openai_size,
            quality='medium',
            style='pixel art style' if 'pixel' in description.lower() else None,
            transparent_background=True,
            tags=auto_tags
        )
    
    def generate_tileset(
        self,
        tile_description: str,
        count: int = 1,
        filename_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Generate a set of tiles with consistent style.
        
        Args:
            tile_description: Description of the tiles
            count: Number of tile variations to generate
            filename_prefix: Prefix for generated filenames
            tags: Searchable tags
            
        Returns:
            List of asset information dicts
        """
        results = []
        
        base_prompt = f"tileable game tile, {tile_description}, seamless pattern"
        
        # Auto-generate tags
        auto_tags = ['tile', 'tileset']
        if tags:
            auto_tags.extend(tags)
        
        for i in range(count):
            filename = f"{filename_prefix}_{i+1}.png" if filename_prefix else None
            
            result = self.generate(
                prompt=base_prompt,
                category='tiles',
                filename=filename,
                size='1024x1024',
                quality='medium',
                style='pixel art style' if 'pixel' in tile_description.lower() else None,
                tags=auto_tags
            )
            
            results.append(result)
        
        return results
    
    def generate_ui_element(
        self,
        element_type: str,
        style_description: str,
        filename: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate a UI element (button, icon, health bar, etc.).
        
        Args:
            element_type: Type of UI element
            style_description: Style and color details
            filename: Output filename
            tags: Searchable tags
            
        Returns:
            Asset information dict
        """
        prompt = f"game UI {element_type}, {style_description}, clean design, modern"
        
        # Auto-generate tags
        auto_tags = ['ui', element_type.lower()]
        if tags:
            auto_tags.extend(tags)
        
        return self.generate(
            prompt=prompt,
            category='ui',
            filename=filename,
            size='1024x1024',
            quality='medium',
            transparent_background=True,
            tags=auto_tags
        )
    
    def search_library(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search the asset library.
        
        Args:
            query: Text search query
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of matching assets
        """
        return self.library.search_assets(query, category, tags)
    
    def get_total_cost(self) -> float:
        """Get total cost of all generations."""
        return self.library.get_total_cost()
    
    def get_library_stats(self) -> Dict:
        """Get statistics about the asset library."""
        assets = self.library.index['assets']
        
        stats = {
            'total_assets': len(assets),
            'total_cost': self.library.get_total_cost(),
            'by_category': {},
            'recent_generations': self.library.cost_data.get('generations', [])[-5:]
        }
        
        for asset in assets:
            cat = asset['category']
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
        
        return stats
    
    def _closest_openai_size(self, size: str) -> str:
        """Convert arbitrary size string to closest OpenAI size."""
        try:
            w, h = map(int, size.lower().replace('x', ' ').split())
            
            if w > h:
                return '1792x1024'
            elif h > w:
                return '1024x1792'
            else:
                return '1024x1024'
        except:
            return '1024x1024'


def main():
    """Example usage / CLI interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_generator.py <prompt> [category] [filename]")
        print("\nCategories: sprites, tiles, ui, backgrounds, effects")
        print("\nExample:")
        print("  python image_generator.py 'a blue robot character' sprites robot.png")
        sys.exit(1)
    
    prompt = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else 'sprites'
    filename = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Initialize generator
    generator = ImageGenerator()
    
    # Generate image
    result = generator.generate(prompt, category=category, filename=filename)
    
    print(f"\n✓ Success!")
    print(f"  File: {result['filename']}")
    print(f"  Path: {result['full_path']}")
    print(f"  Cost: ${result['cost']:.2f}")
    print(f"  Total library cost: ${generator.get_total_cost():.2f}")


if __name__ == '__main__':
    main()
