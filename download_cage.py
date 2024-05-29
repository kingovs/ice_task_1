from bing_image_downloader import downloader

def download_images(query, limit, output_dir):
    # Download images using bing-image-downloader
    try:
        downloader.download(query, limit=limit, output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=60)
        print(f"Downloaded images for query '{query}' to directory: {output_dir}/{query}")
    except Exception as e:
        print(f"An error occurred while downloading images for query '{query}': {e}")

if __name__ == "__main__":
    # Download images for specified queries
    download_images("Nicolas Cage Face", 100, "data/cage")
    download_images("Men's Faces", 100, "data/others")
