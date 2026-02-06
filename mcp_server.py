#!/usr/bin/env python3
"""
Godot Image Generator MCP Server
Exposes image generation capabilities through MCP protocol
"""

import asyncio
import logging
from typing import Optional, List
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import Field

from image_generator import ImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("godot-image-gen-mcp")

# Initialize the MCP server
app = Server("godot-image-gen")

# Initialize the image generator (will be done in main)
generator: Optional[ImageGenerator] = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="generate_sprite",
            description=(
                "Generate a game sprite with transparent background. "
                "Automatically saves to the shared asset library and catalogs for reuse. "
                "Use this for characters, enemies, items, or any sprite assets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the sprite to generate (e.g., 'robot character with blue armor')"
                    },
                    "subcategory": {
                        "type": "string",
                        "description": "Sprite subcategory: 'characters', 'enemies', or 'items'",
                        "enum": ["characters", "enemies", "items"],
                        "default": "characters"
                    },
                    "size": {
                        "type": "string",
                        "description": "Sprite dimensions (e.g., '32x32', '64x64', '256x256')",
                        "default": "256x256"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (auto-generated if not provided)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional searchable tags for cataloging"
                    }
                },
                "required": ["description"]
            }
        ),
        Tool(
            name="generate_tileset",
            description=(
                "Generate a set of seamless tileable textures for game levels. "
                "Creates multiple variations with consistent style. "
                "Perfect for terrain, walls, floors, and other repeating patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tile_description": {
                        "type": "string",
                        "description": "Description of the tiles (e.g., 'grass and dirt terrain')"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of tile variations to generate",
                        "default": 4,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "filename_prefix": {
                        "type": "string",
                        "description": "Optional prefix for generated filenames"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional searchable tags"
                    }
                },
                "required": ["tile_description"]
            }
        ),
        Tool(
            name="generate_ui_element",
            description=(
                "Generate a UI element like buttons, icons, health bars, or HUD elements. "
                "Includes transparent background for easy integration into UI."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "element_type": {
                        "type": "string",
                        "description": "Type of UI element (e.g., 'button', 'icon', 'health bar', 'progress bar')"
                    },
                    "style_description": {
                        "type": "string",
                        "description": "Style and appearance details (e.g., 'red pause button with rounded corners')"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional searchable tags"
                    }
                },
                "required": ["element_type", "style_description"]
            }
        ),
        Tool(
            name="generate_custom_image",
            description=(
                "Generate a custom image with full control over all parameters. "
                "Use this for backgrounds, effects, or any asset that doesn't fit other categories. "
                "Supports custom sizes, quality levels, and styling."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate"
                    },
                    "category": {
                        "type": "string",
                        "description": "Asset category (sprites, tiles, ui, backgrounds, effects)",
                        "enum": ["sprites", "tiles", "ui", "backgrounds", "effects"],
                        "default": "sprites"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename"
                    },
                    "size": {
                        "type": "string",
                        "description": "Image dimensions: '1024x1024', '1792x1024', or '1024x1792'",
                        "enum": ["1024x1024", "1792x1024", "1024x1792"],
                        "default": "1024x1024"
                    },
                    "quality": {
                        "type": "string",
                        "description": "Generation quality (affects cost): 'low' ($0.01), 'medium' ($0.04), 'high' ($0.17)",
                        "enum": ["low", "medium", "high"],
                        "default": "medium"
                    },
                    "style": {
                        "type": "string",
                        "description": "Optional style modifier (e.g., 'pixel art', 'hand-drawn', '3D rendered')"
                    },
                    "transparent_background": {
                        "type": "boolean",
                        "description": "Whether to request transparent background",
                        "default": False
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional searchable tags"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="search_asset_library",
            description=(
                "Search the shared asset library before generating new assets. "
                "This helps avoid duplicate generations and saves costs. "
                "Search by keywords, category, or tags to find existing assets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text search query (searches prompts, filenames, and tags)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (e.g., 'sprites', 'sprites/characters', 'tiles', 'ui')"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (returns assets matching any tag)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_library_stats",
            description=(
                "Get statistics about the shared asset library. "
                "Shows total assets, cost tracking, breakdown by category, and recent generations. "
                "Useful for understanding library growth and spending."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_total_cost",
            description=(
                "Get the total cost of all image generations across all projects. "
                "Tracks cumulative spending on the OpenAI API."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    try:
        if name == "generate_sprite":
            result = generator.generate_sprite(
                description=arguments["description"],
                subcategory=arguments.get("subcategory", "characters"),
                size=arguments.get("size", "256x256"),
                filename=arguments.get("filename"),
                tags=arguments.get("tags")
            )
            
            return [
                TextContent(
                    type="text",
                    text=f"✓ Sprite generated successfully!\n\n"
                         f"**File:** {result['filename']}\n"
                         f"**Path:** {result['full_path']}\n"
                         f"**Category:** {result['category']}\n"
                         f"**Cost:** ${result['cost']:.2f}\n"
                         f"**Total Library Cost:** ${generator.get_total_cost():.2f}\n\n"
                         f"The sprite has been saved to the shared asset library and cataloged for reuse."
                )
            ]
        
        elif name == "generate_tileset":
            results = generator.generate_tileset(
                tile_description=arguments["tile_description"],
                count=arguments.get("count", 4),
                filename_prefix=arguments.get("filename_prefix"),
                tags=arguments.get("tags")
            )
            
            total_cost = sum(r['cost'] for r in results)
            file_list = "\n".join([f"  - {r['filename']}" for r in results])
            
            return [
                TextContent(
                    type="text",
                    text=f"✓ Tileset generated successfully!\n\n"
                         f"**Generated {len(results)} tiles:**\n{file_list}\n\n"
                         f"**Cost:** ${total_cost:.2f}\n"
                         f"**Total Library Cost:** ${generator.get_total_cost():.2f}\n\n"
                         f"All tiles have been saved to the shared asset library."
                )
            ]
        
        elif name == "generate_ui_element":
            result = generator.generate_ui_element(
                element_type=arguments["element_type"],
                style_description=arguments["style_description"],
                filename=arguments.get("filename"),
                tags=arguments.get("tags")
            )
            
            return [
                TextContent(
                    type="text",
                    text=f"✓ UI element generated successfully!\n\n"
                         f"**File:** {result['filename']}\n"
                         f"**Path:** {result['full_path']}\n"
                         f"**Type:** {arguments['element_type']}\n"
                         f"**Cost:** ${result['cost']:.2f}\n"
                         f"**Total Library Cost:** ${generator.get_total_cost():.2f}"
                )
            ]
        
        elif name == "generate_custom_image":
            result = generator.generate(
                prompt=arguments["prompt"],
                category=arguments.get("category", "sprites"),
                filename=arguments.get("filename"),
                size=arguments.get("size", "1024x1024"),
                quality=arguments.get("quality", "medium"),
                style=arguments.get("style"),
                transparent_background=arguments.get("transparent_background", False),
                tags=arguments.get("tags")
            )
            
            return [
                TextContent(
                    type="text",
                    text=f"✓ Custom image generated successfully!\n\n"
                         f"**File:** {result['filename']}\n"
                         f"**Path:** {result['full_path']}\n"
                         f"**Category:** {result['category']}\n"
                         f"**Size:** {arguments.get('size', '1024x1024')}\n"
                         f"**Quality:** {arguments.get('quality', 'medium')}\n"
                         f"**Cost:** ${result['cost']:.2f}\n"
                         f"**Total Library Cost:** ${generator.get_total_cost():.2f}"
                )
            ]
        
        elif name == "search_asset_library":
            results = generator.search_library(
                query=arguments.get("query"),
                category=arguments.get("category"),
                tags=arguments.get("tags")
            )
            
            if not results:
                return [
                    TextContent(
                        type="text",
                        text="No assets found matching your search criteria.\n\n"
                             "You may want to generate a new asset instead."
                    )
                ]
            
            # Format results
            result_text = f"Found {len(results)} asset(s) in the library:\n\n"
            for i, asset in enumerate(results[:20], 1):  # Limit to 20 results
                tags_str = ", ".join(asset.get('tags', [])) if asset.get('tags') else "none"
                result_text += f"**{i}. {asset['filename']}**\n"
                result_text += f"   - Path: `{asset['full_path']}`\n"
                result_text += f"   - Category: {asset['category']}\n"
                result_text += f"   - Prompt: {asset['prompt'][:80]}...\n"
                result_text += f"   - Tags: {tags_str}\n"
                result_text += f"   - Created: {asset['created'][:10]}\n\n"
            
            if len(results) > 20:
                result_text += f"\n...and {len(results) - 20} more results."
            
            return [TextContent(type="text", text=result_text)]
        
        elif name == "get_library_stats":
            stats = generator.get_library_stats()
            
            # Format category breakdown
            cat_breakdown = "\n".join([
                f"  - {cat}: {count} assets"
                for cat, count in stats['by_category'].items()
            ])
            
            # Format recent generations
            recent_text = ""
            if stats['recent_generations']:
                recent_text = "\n\n**Recent Generations:**\n"
                for gen in stats['recent_generations']:
                    recent_text += f"  - {gen['prompt'][:60]}... (${gen['cost']:.2f})\n"
            
            return [
                TextContent(
                    type="text",
                    text=f"📊 **Asset Library Statistics**\n\n"
                         f"**Total Assets:** {stats['total_assets']}\n"
                         f"**Total Cost:** ${stats['total_cost']:.2f}\n\n"
                         f"**Breakdown by Category:**\n{cat_breakdown}"
                         f"{recent_text}"
                )
            ]
        
        elif name == "get_total_cost":
            total_cost = generator.get_total_cost()
            return [
                TextContent(
                    type="text",
                    text=f"💰 **Total Cost:** ${total_cost:.2f}\n\n"
                         f"This represents all image generation costs across all projects using this library."
                )
            ]
        
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )
            ]
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [
            TextContent(
                type="text",
                text=f"❌ Error: {str(e)}\n\n"
                     f"Please check the logs for more details."
            )
        ]


async def main():
    """Run the MCP server."""
    global generator
    
    # Initialize the image generator
    try:
        logger.info("Initializing Godot Image Generator...")
        generator = ImageGenerator()
        logger.info(f"✓ Asset library initialized at: {generator.library.library_dir}")
        logger.info(f"✓ Total library cost: ${generator.get_total_cost():.2f}")
    except Exception as e:
        logger.error(f"Failed to initialize image generator: {e}")
        raise
    
    # Run the MCP server
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Godot Image Generator MCP Server started")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
