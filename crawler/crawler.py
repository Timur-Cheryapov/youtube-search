import yt_dlp
import json
from typing import List, Dict

def get_video_urls_from_channel(url: str, limit: int = 50) -> List[str]:
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
        'playlistend': limit,  # Limit number of videos per playlist
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        save_to_json(info, "./crawler/youtube_info.json")

        channel_url = info.get('channel_url')
        
        video_urls = []
        entries = info.get('entries', [])
        
        # Handle channel URLs which have nested playlists (Videos, Shorts, etc.)
        for entry in entries:
            if entry.get('_type') == 'playlist' and 'entries' in entry:
                # This is a sub-playlist (like "Videos" or "Shorts")
                sub_entries = entry.get('entries', [])
                for video_entry in sub_entries[:limit]:  # Apply limit per playlist
                    if video_entry.get('_type') == 'url' and video_entry.get('ie_key') == 'Youtube':
                        video_urls.append(f"https://www.youtube.com/watch?v={video_entry['id']}")
            elif entry.get('_type') == 'url' and entry.get('ie_key') == 'Youtube':
                # Direct video entry (for regular playlists)
                video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
        
        return {
            "video_urls": video_urls[:limit], # Apply overall limit
            "channel_url": channel_url
        }

def fetch_video_metadata(video_url: str) -> Dict:
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "url": f"https://www.youtube.com/watch?v={info.get('id')}",
            "description": info.get("description"),
            "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "channel_id": info.get("channel_id"),
            "channel": info.get("channel"),
            "thumbnails": info.get("thumbnails"),
        }

def save_to_json(data: Dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    url = input("Enter YouTube channel URL: ").strip()
    limit = 100
    data = get_video_urls_from_channel(url, limit)
    channel_url = data['channel_url']
    print(f"Found {len(data['video_urls'])} videos. Fetching metadata...")
    results = []
    for idx, video_url in enumerate(data['video_urls'], 1):
        print(f"Fetching {idx}/{len(data['video_urls'])}: {video_url}")
        try:
            results.append(fetch_video_metadata(video_url))
        except Exception as e:
            print(f"Failed to fetch {video_url}: {e}")
    output = {channel_url: results}
    save_to_json(output, "./crawler/youtube_videos.json")
    print(f"Saved {len(results)} videos to youtube_videos.json under key: {channel_url}")