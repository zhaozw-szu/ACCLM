#!/usr/bin/env python3
"""
AVI Container Parser
Parses AVI/RIFF files and creates detailed semantic tree
"""

import struct
import json
import os
from typing import Dict, List, Any, Optional
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from get_container import help_fun_v4


class AVIParser:
    """AVI/RIFF container parser"""
    
    # FourCC codes
    FOURCC_NAMES = {
        b'RIFF': 'Resource Interchange File Format',
        b'AVI ': 'Audio Video Interleave',
        b'WAVE': 'Waveform Audio',
        b'LIST': 'List Container',
        b'hdrl': 'Header List',
        b'strl': 'Stream List',
        b'movi': 'Movie Data',
        b'idx1': 'Index',
        b'avih': 'AVI Header',
        b'strh': 'Stream Header',
        b'strf': 'Stream Format',
        b'strn': 'Stream Name',
        b'JUNK': 'Junk/Padding',
        b'INFO': 'Info List',
        b'ISFT': 'Software',
        b'IDIT': 'Creation Date',
        b'INAM': 'Name/Title',
        b'IART': 'Artist',
        b'ICMT': 'Comment',
    }
    
    # Stream types
    STREAM_TYPES = {
        b'vids': 'Video Stream',
        b'auds': 'Audio Stream',
        b'txts': 'Text Stream',
        b'mids': 'MIDI Stream',
    }
    
    # Video codecs
    VIDEO_CODECS = {
        b'DIB ': 'Uncompressed RGB',
        b'RGB ': 'Uncompressed RGB',
        b'MJPG': 'Motion JPEG',
        b'mjpg': 'Motion JPEG',
        b'H264': 'H.264/AVC',
        b'h264': 'H.264/AVC',
        b'X264': 'H.264/AVC (x264)',
        b'x264': 'H.264/AVC (x264)',
        b'AVC1': 'H.264/AVC',
        b'avc1': 'H.264/AVC',
        b'XVID': 'Xvid MPEG-4',
        b'xvid': 'Xvid MPEG-4',
        b'DIVX': 'DivX MPEG-4',
        b'divx': 'DivX MPEG-4',
        b'DX50': 'DivX 5',
        b'MP4V': 'MPEG-4 Video',
        b'mp4v': 'MPEG-4 Video',
        b'FMP4': 'FFmpeg MPEG-4',
        b'WMV3': 'Windows Media Video 9',
        b'WMV2': 'Windows Media Video 8',
        b'WMV1': 'Windows Media Video 7',
    }
    
    # Audio formats
    AUDIO_FORMATS = {
        0x0001: 'PCM',
        0x0002: 'MS ADPCM',
        0x0003: 'IEEE Float',
        0x0006: 'ITU G.711 A-law',
        0x0007: 'ITU G.711 µ-law',
        0x0011: 'IMA ADPCM',
        0x0016: 'ITU G.723 ADPCM',
        0x0031: 'GSM 6.10',
        0x0050: 'MPEG Audio',
        0x0055: 'MP3',
        0x0160: 'WMA',
        0x0161: 'WMA 2',
        0x0162: 'WMA Pro',
        0x0163: 'WMA Lossless',
        0x2000: 'AC-3',
        0x2001: 'DTS',
    }
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.file = None
        self.frames = []  # Store reconstructed frame information
        
    def parse(self) -> Dict[str, Any]:
        """Parse the AVI file and return semantic tree"""
        with open(self.filepath, 'rb') as f:
            self.file = f
            
            # Read RIFF header
            riff_id = f.read(4)
            if riff_id != b'RIFF':
                print(self.filepath)
                raise ValueError("Not a valid RIFF file")
            
            file_size = struct.unpack('<I', f.read(4))[0]
            file_type = f.read(4)
            
            if file_type != b'AVI ':
                raise ValueError(f"Not an AVI file (type: {file_type})")
            
            # Parse all chunks as flat array like MP4Box.js
            result = []
            
            # Add file info
            file_info = {
                'type': 'RIFF',
                'format': 'Audio Video Interleave',
                'size': file_size,
                'file_type': file_type.decode('ascii', errors='ignore'),
                'hdr_size': 12,
                'start': 0
            }
            
            # Analyze file type for hints
            if file_type == b'AVI ':
                file_info['format_standard'] = 'Standard AVI'
            elif file_type == b'AVIX':
                file_info['format_standard'] = 'Extended AVI (OpenDML)'
                file_info['size_hint'] = 'Large file (>1GB)'
            
            result.append(file_info)
            
            # Parse all chunks
            while f.tell() < min(file_size + 8, self.file_size):
                try:
                    chunk = self._parse_chunk(f.tell())
                    if chunk:
                        result.append(chunk)
                    else:
                        break
                except Exception as e:
                    print(f"Error parsing chunk at {f.tell()}: {e}")
                    break
            
            # Add reconstructed frames (like samples in MP4)
            for frame in self.frames:
                result.append(frame)
            
            self.file = None
            return result
    
    def _parse_chunk(self, offset: int) -> Optional[Dict[str, Any]]:
        """Parse a single RIFF chunk"""
        if offset >= self.file_size:
            return None
        
        self.file.seek(offset)
        
        # Read chunk header
        fourcc = self.file.read(4)
        if len(fourcc) < 4:
            return None
        
        size = struct.unpack('<I', self.file.read(4))[0]
        
        chunk = {
            'type': fourcc.decode('ascii', errors='ignore'),
            'size': size,
            'hdr_size': 8,
            'start': offset
        }
        
        # Parse specific chunks
        if fourcc == b'LIST':
            list_type = self.file.read(4)
            chunk['list_type'] = list_type.decode('ascii', errors='ignore')
            chunk['hdr_size'] = 12  # LIST has extra 4 bytes for type
            
            # Parse list contents
            if list_type in [b'hdrl', b'strl', b'movi', b'INFO']:
                chunk['chunks'] = []
                current_offset = offset + 12
                end_offset = offset + 8 + size
                
                while current_offset < end_offset:
                    child = self._parse_chunk(current_offset)
                    if child:
                        chunk['chunks'].append(child)
                        # Size + 8 for header, aligned to 2 bytes
                        child_size = child['size'] + 8
                        if child_size % 2:
                            child_size += 1
                        current_offset += child_size
                    else:
                        break
        
        elif fourcc == b'avih':
            chunk['data'] = self._parse_avih()
        
        elif fourcc == b'strh':
            chunk['data'] = self._parse_strh()
        
        elif fourcc == b'strf':
            # Stream format depends on stream type
            # We need context from parent strh, so we'll parse it generically
            chunk['data'] = self._parse_strf(size)
        
        elif fourcc == b'strn':
            # Stream name - often contains device/software info
            name = self.file.read(size).rstrip(b'\x00').decode('utf-8', errors='ignore')
            chunk['name'] = name
            chunk['stream_name'] = name
            
            # Device/software detection from stream name
            if name:
                name_upper = name.upper()
                if any(x in name_upper for x in ['FUJI', 'FUJIFILM']):
                    chunk['device_hint'] = 'Fujifilm'
                elif any(x in name_upper for x in ['CANON', 'EOS']):
                    chunk['device_hint'] = 'Canon'
                elif any(x in name_upper for x in ['SONY', 'XAVC']):
                    chunk['device_hint'] = 'Sony'
                elif any(x in name_upper for x in ['NIKON']):
                    chunk['device_hint'] = 'Nikon'
                elif any(x in name_upper for x in ['PANASONIC', 'LUMIX']):
                    chunk['device_hint'] = 'Panasonic'
                elif any(x in name_upper for x in ['GOPRO']):
                    chunk['device_hint'] = 'GoPro'
                elif any(x in name_upper for x in ['DJI', 'OSMO', 'MAVIC']):
                    chunk['device_hint'] = 'DJI'
                elif any(x in name_upper for x in ['PREMIERE', 'AFTEREFFECTS', 'ENCODER']):
                    chunk['software_hint'] = 'Adobe'
                elif any(x in name_upper for x in ['VEGAS', 'MOVIE STUDIO']):
                    chunk['software_hint'] = 'Sony Vegas'
                elif any(x in name_upper for x in ['FFMPEG', 'LIBAV']):
                    chunk['software_hint'] = 'FFmpeg'
                elif any(x in name_upper for x in ['VIRTUALDUB']):
                    chunk['software_hint'] = 'VirtualDub'
        
        elif fourcc == b'idx1':
            chunk['data'] = self._parse_idx1(size)
        
        elif fourcc in [b'ISFT', b'INAM', b'IART', b'ICMT', b'IDIT', b'ISRC', b'ISBJ', b'ICOP']:
            # Info strings - metadata
            text = self.file.read(size).rstrip(b'\x00').decode('utf-8', errors='ignore')
            chunk['text'] = text
            
            # Specific field names
            info_names = {
                b'ISFT': ('software', 'Software/Encoder'),
                b'INAM': ('name', 'Title/Name'),
                b'IART': ('artist', 'Artist/Author'),
                b'ICMT': ('comment', 'Comment'),
                b'IDIT': ('creation_date', 'Creation Date'),
                b'ISRC': ('source', 'Source'),
                b'ISBJ': ('subject', 'Subject'),
                b'ICOP': ('copyright', 'Copyright')
            }
            
            if fourcc in info_names:
                field_name, description = info_names[fourcc]
                chunk['field_name'] = field_name
                chunk['field_description'] = description
            
            # Software detection from ISFT
            if fourcc == b'ISFT' and text:
                text_upper = text.upper()
                if any(x in text_upper for x in ['PREMIERE', 'AFTEREFFECTS', 'ENCODER', 'ADOBE']):
                    chunk['software_detected'] = 'Adobe'
                elif any(x in text_upper for x in ['VEGAS', 'MOVIE STUDIO']):
                    chunk['software_detected'] = 'Sony Vegas'
                elif any(x in text_upper for x in ['FFMPEG', 'LIBAV']):
                    chunk['software_detected'] = 'FFmpeg'
                elif any(x in text_upper for x in ['VIRTUALDUB']):
                    chunk['software_detected'] = 'VirtualDub'
                elif any(x in text_upper for x in ['AVIDEMUX']):
                    chunk['software_detected'] = 'Avidemux'
                elif any(x in text_upper for x in ['HANDBRAKE']):
                    chunk['software_detected'] = 'HandBrake'
                elif any(x in text_upper for x in ['CAMTASIA']):
                    chunk['software_detected'] = 'Camtasia'
        
        elif fourcc == b'JUNK':
            chunk['is_padding'] = True
        
        elif size > 0 and size < 1024 * 1024:  # Don't read huge data chunks
            # Store small unknown chunks
            if fourcc[:2] == b'00' or fourcc[:2] == b'01':  # Video/audio data
                chunk['is_media_data'] = True
            else:
                data = self.file.read(min(size, 1024))
                if len(data) <= 256:
                    chunk['raw_data'] = list(data)
        
        # Seek to next chunk (aligned to 2 bytes)
        next_offset = offset + 8 + size
        if size % 2:
            next_offset += 1
        self.file.seek(next_offset)
        
        return chunk
    
    def _parse_avih(self) -> Dict[str, Any]:
        """Parse AVI Main Header"""
        data = {}
        
        data['microsec_per_frame'] = struct.unpack('<I', self.file.read(4))[0]
        data['max_bytes_per_sec'] = struct.unpack('<I', self.file.read(4))[0]
        data['padding_granularity'] = struct.unpack('<I', self.file.read(4))[0]
        
        flags = struct.unpack('<I', self.file.read(4))[0]
        data['flags'] = flags
        data['flags_decoded'] = {
            'has_index': bool(flags & 0x10),
            'must_use_index': bool(flags & 0x20),
            'is_interleaved': bool(flags & 0x100),
            'trust_ck_type': bool(flags & 0x800),
            'was_capture_file': bool(flags & 0x10000),
            'copyrighted': bool(flags & 0x20000)
        }
        
        # Analyze flags for device origin
        if data['flags_decoded']['was_capture_file']:
            data['origin_hint'] = 'Direct capture from camera/capture card'
        else:
            data['origin_hint'] = 'Edited or transcoded file'
        
        if data['flags_decoded']['is_interleaved']:
            data['interleave_hint'] = 'Optimized for streaming playback'
        
        data['total_frames'] = struct.unpack('<I', self.file.read(4))[0]
        data['initial_frames'] = struct.unpack('<I', self.file.read(4))[0]
        data['streams'] = struct.unpack('<I', self.file.read(4))[0]
        data['suggested_buffer_size'] = struct.unpack('<I', self.file.read(4))[0]
        
        data['width'] = struct.unpack('<I', self.file.read(4))[0]
        data['height'] = struct.unpack('<I', self.file.read(4))[0]
        
        # Reserved
        data['reserved'] = list(struct.unpack('<4I', self.file.read(16)))
        
        # Calculate frame rate
        if data['microsec_per_frame'] > 0:
            data['frame_rate'] = 1000000.0 / data['microsec_per_frame']
        
        # Calculate duration
        if data['microsec_per_frame'] > 0 and data['total_frames'] > 0:
            data['duration_seconds'] = (data['total_frames'] * data['microsec_per_frame']) / 1000000.0
        
        return data
    
    def _parse_strh(self) -> Dict[str, Any]:
        """Parse Stream Header"""
        data = {}
        
        fcc_type = self.file.read(4)
        data['stream_type'] = fcc_type.decode('ascii', errors='ignore')
        data['stream_type_name'] = self.STREAM_TYPES.get(fcc_type, 'Unknown')
        
        fcc_handler = self.file.read(4)
        data['codec'] = fcc_handler.decode('ascii', errors='ignore')
        
        if fcc_type == b'vids':
            data['codec_name'] = self.VIDEO_CODECS.get(fcc_handler, f'Unknown ({data["codec"]})')
        elif fcc_type == b'auds':
            data['codec_name'] = 'See stream format for details'
        
        flags = struct.unpack('<I', self.file.read(4))[0]
        data['flags'] = flags
        
        data['priority'] = struct.unpack('<H', self.file.read(2))[0]
        data['language'] = struct.unpack('<H', self.file.read(2))[0]
        data['initial_frames'] = struct.unpack('<I', self.file.read(4))[0]
        scale = struct.unpack('<I', self.file.read(4))[0]
        rate = struct.unpack('<I', self.file.read(4))[0]
        data['scale'] = scale
        data['rate'] = rate
        
        frame_rate = 0
        if scale > 0:
            frame_rate = rate / scale
            data['frame_rate'] = frame_rate
            
            # Frame rate analysis
            if fcc_type == b'vids' and frame_rate > 0:
                if 23.9 <= frame_rate <= 24.1:
                    data['frame_rate_hint'] = 'Film standard (24fps)'
                elif 24.9 <= frame_rate <= 25.1:
                    data['frame_rate_hint'] = 'PAL standard (25fps)'
                elif 29.9 <= frame_rate <= 30.1:
                    data['frame_rate_hint'] = 'NTSC standard (30fps)'
                elif 59.9 <= frame_rate <= 60.1:
                    data['frame_rate_hint'] = 'High frame rate (60fps)'
        
        data['start'] = struct.unpack('<I', self.file.read(4))[0]
        data['length'] = struct.unpack('<I', self.file.read(4))[0]
        
        if data['scale'] > 0 and data['rate'] > 0:
            data['duration_seconds'] = (data['length'] * data['scale']) / data['rate']
        
        data['suggested_buffer_size'] = struct.unpack('<I', self.file.read(4))[0]
        quality = struct.unpack('<I', self.file.read(4))[0]
        data['quality'] = quality
        data['sample_size'] = struct.unpack('<I', self.file.read(4))[0]
        
        # Analyze quality for device hints
        if quality == 10000:
            data['quality_hint'] = 'Maximum quality (typical for capture devices)'
        elif quality == 0xFFFFFFFF or quality == -1:
            data['quality_hint'] = 'Default quality (typical for encoders)'
        
        # Rectangle
        left = struct.unpack('<h', self.file.read(2))[0]
        top = struct.unpack('<h', self.file.read(2))[0]
        right = struct.unpack('<h', self.file.read(2))[0]
        bottom = struct.unpack('<h', self.file.read(2))[0]
        data['frame'] = {
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'width': right - left,
            'height': bottom - top
        }
        
        return data
    
    def _parse_strf(self, size: int) -> Dict[str, Any]:
        """Parse Stream Format"""
        # Try to parse as video format (BITMAPINFOHEADER)
        if size >= 40:
            start_pos = self.file.tell()
            
            biSize = struct.unpack('<I', self.file.read(4))[0]
            
            if biSize == 40:  # BITMAPINFOHEADER
                data = {'format_type': 'BITMAPINFOHEADER'}
                
                data['width'] = struct.unpack('<i', self.file.read(4))[0]
                data['height'] = struct.unpack('<i', self.file.read(4))[0]
                data['planes'] = struct.unpack('<H', self.file.read(2))[0]
                data['bit_count'] = struct.unpack('<H', self.file.read(2))[0]
                
                compression = self.file.read(4)
                data['compression'] = compression.decode('ascii', errors='ignore')
                data['compression_name'] = self.VIDEO_CODECS.get(compression, 'Unknown')
                data['compression_fourcc'] = compression.hex()
                
                # Codec-based device hints
                if compression in [b'MJPG', b'mjpg']:
                    data['codec_hint'] = 'Digital camera or webcam (Motion JPEG common in cameras)'
                elif compression in [b'H264', b'h264', b'avc1', b'X264', b'x264']:
                    data['codec_hint'] = 'Modern encoder or camera (H.264)'
                elif compression in [b'XVID', b'xvid', b'DX50', b'DIVX']:
                    data['codec_hint'] = 'Software encoder (MPEG-4 ASP)'
                elif compression == b'YUY2':
                    data['codec_hint'] = 'Uncompressed or capture device'
                elif compression == b'dvsd':
                    data['codec_hint'] = 'DV camcorder'
                
                data['size_image'] = struct.unpack('<I', self.file.read(4))[0]
                data['x_pels_per_meter'] = struct.unpack('<i', self.file.read(4))[0]
                data['y_pels_per_meter'] = struct.unpack('<i', self.file.read(4))[0]
                data['clr_used'] = struct.unpack('<I', self.file.read(4))[0]
                data['clr_important'] = struct.unpack('<I', self.file.read(4))[0]
                
                # Read any extra data
                remaining = size - (self.file.tell() - start_pos)
                if remaining > 0 and remaining < 1024:
                    extra_data = self.file.read(remaining)
                    if len(extra_data) > 0:
                        data['extra_data'] = list(extra_data[:256])
                
                return data
        
        # Try to parse as audio format (WAVEFORMATEX)
        if size >= 16:
            start_pos = self.file.tell()
            
            format_tag = struct.unpack('<H', self.file.read(2))[0]
            data = {
                'format_type': 'WAVEFORMATEX',
                'format_tag': format_tag,
                'format_name': self.AUDIO_FORMATS.get(format_tag, f'Unknown (0x{format_tag:04X})')
            }
            
            channels = struct.unpack('<H', self.file.read(2))[0]
            sample_rate = struct.unpack('<I', self.file.read(4))[0]
            data['channels'] = channels
            data['samples_per_sec'] = sample_rate
            data['avg_bytes_per_sec'] = struct.unpack('<I', self.file.read(4))[0]
            data['block_align'] = struct.unpack('<H', self.file.read(2))[0]
            data['bits_per_sample'] = struct.unpack('<H', self.file.read(2))[0]
            
            # Audio format hints
            if format_tag == 0x0001:  # PCM
                if sample_rate == 48000:
                    data['audio_hint'] = 'Professional audio (48kHz)'
                elif sample_rate == 44100:
                    data['audio_hint'] = 'Consumer audio (CD quality)'
                if channels == 1:
                    data['channel_hint'] = 'Mono (typical for voice recording)'
                elif channels == 2:
                    data['channel_hint'] = 'Stereo (standard)'
            elif format_tag == 0x0055:  # MP3
                data['audio_hint'] = 'Compressed with MP3 encoder'
            
            if size > 16:
                cb_size = struct.unpack('<H', self.file.read(2))[0]
                data['cb_size'] = cb_size
                
                if cb_size > 0 and cb_size < 1024:
                    extra_data = self.file.read(cb_size)
                    if len(extra_data) > 0:
                        data['extra_data'] = list(extra_data[:256])
            
            return data
        
        # Unknown format
        data = self.file.read(min(size, 256))
        return {
            'format_type': 'unknown',
            'raw_data': list(data)
        }
    
    def _parse_idx1(self, size: int) -> Dict[str, Any]:
        """Parse Index and reconstruct frame information"""
        entry_count = size // 16
        
        data = {
            'entry_count': entry_count
        }
        
        # Read and reconstruct all entries as frames (like MP4 samples)
        frame_number = 0
        dts = 0
        
        for i in range(entry_count):
            chunk_id = self.file.read(4).decode('ascii', errors='ignore')
            flags = struct.unpack('<I', self.file.read(4))[0]
            offset = struct.unpack('<I', self.file.read(4))[0]
            length = struct.unpack('<I', self.file.read(4))[0]
            
            # Determine stream type from chunk_id
            stream_number = None
            data_type = None
            if len(chunk_id) >= 2:
                try:
                    stream_number = int(chunk_id[:2], 10)
                    data_type = chunk_id[2:]
                except:
                    pass
            
            # Create frame entry (similar to MP4 sample)
            frame = {
                'type': 'frame',
                'number': frame_number,
                'chunk_id': chunk_id,
                'stream_number': stream_number,
                'data_type': data_type,
                'is_video': data_type in ['dc', 'db'],
                'is_audio': data_type in ['wb'],
                'is_keyframe': bool(flags & 0x10),
                'flags': flags,
                'offset': offset,
                'length': length,
                'dts': dts
            }
            
            self.frames.append(frame)
            frame_number += 1
            
            # Increment DTS (simplified, real calculation needs frame rate)
            if data_type in ['dc', 'db']:  # Video frames
                dts += 1
        
        return data


def format_bytes(bytes_value: int) -> str:
    """Format byte size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def generate_summary(tree: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of the AVI file"""
    # Extract file info
    file_info_chunk = next((item for item in tree if item.get('type') == 'RIFF'), {})
    file_size = file_info_chunk.get('size', 0) + 8
    
    summary = {
        'file_size': file_size,
        'streams': [],
        'video_info': {},
        'audio_info': {}
    }
    
    # Find hdrl list
    for chunk in tree:
        if chunk.get('type') == 'LIST' and chunk.get('list_type') == 'hdrl':
            # Find avih
            for child in chunk.get('chunks', []):
                if child.get('type') == 'avih' and 'data' in child:
                    data = child['data']
                    summary['video_info'] = {
                        'width': data.get('width'),
                        'height': data.get('height'),
                        'frame_rate': data.get('frame_rate'),
                        'total_frames': data.get('total_frames'),
                        'duration_seconds': data.get('duration_seconds')
                    }
                
                # Find streams
                elif child.get('type') == 'LIST' and child.get('list_type') == 'strl':
                    stream_info = {}
                    
                    for strl_child in child.get('chunks', []):
                        if strl_child.get('type') == 'strh' and 'data' in strl_child:
                            stream_info.update(strl_child['data'])
                        elif strl_child.get('type') == 'strf' and 'data' in strl_child:
                            stream_info['format'] = strl_child['data']
                        elif strl_child.get('type') == 'strn':
                            stream_info['name'] = strl_child.get('name')
                    
                    summary['streams'].append(stream_info)
                    
                    # Categorize
                    if stream_info.get('stream_type') == 'vids':
                        summary['video_info']['codec'] = stream_info.get('codec_name', 'Unknown')
                    elif stream_info.get('stream_type') == 'auds':
                        fmt = stream_info.get('format', {})
                        summary['audio_info'] = {
                            'format': fmt.get('format_name', 'Unknown'),
                            'channels': fmt.get('channels'),
                            'sample_rate': fmt.get('samples_per_sec'),
                            'bits_per_sample': fmt.get('bits_per_sample')
                        }
    
    return summary



def get_avi_container(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    print(f"Parsing AVI file: {input_file}")
    print(f"File size: {format_bytes(os.path.getsize(input_file))}")
    print()
    
    # Parse the file
    parser = AVIParser(input_file)
    tree = parser.parse()
    
    # Generate summary
    summary = generate_summary(tree)
    
    # Print summary
    print("=== AVI File Summary ===")
    print(f"File: {input_file}")
    print(f"Size: {format_bytes(summary['file_size'])}")
    
    if summary.get('video_info'):
        vi = summary['video_info']
        print(f"\nVideo: {vi.get('codec', 'Unknown')} {vi.get('width', 0)}x{vi.get('height', 0)}")
        if vi.get('frame_rate'):
            print(f"  Frame rate: {vi['frame_rate']:.2f} fps")
        if vi.get('duration_seconds'):
            print(f"  Duration: {vi['duration_seconds']:.2f} seconds")
    
    if summary.get('audio_info') and summary['audio_info'].get('format'):
        ai = summary['audio_info']
        print(f"\nAudio: {ai.get('format', 'Unknown')}")
        if ai.get('sample_rate'):
            print(f"  Sample rate: {ai['sample_rate']} Hz")
        if ai.get('channels'):
            print(f"  Channels: {ai['channels']}")
    
    print(f"\nStreams: {len(summary.get('streams', []))}")
    for stream in summary.get('streams', []):
        print(f"  - {stream.get('stream_type_name', 'Unknown')}: {stream.get('codec_name', 'Unknown')}")
    
    # Count frames
    frames = [item for item in tree if item.get('type') == 'frame']
    if frames:
        video_frames = [f for f in frames if f.get('is_video')]
        audio_frames = [f for f in frames if f.get('is_audio')]
        print(f"\nFrames reconstructed:")
        print(f"  Video frames: {len(video_frames)}")
        print(f"  Audio frames: {len(audio_frames)}")
        print(f"  Total: {len(frames)}")

    return tree


def main(csv_path, save_path):
    df = pd.read_csv(csv_path)
    video_path_list = df["filename"].to_list()
    label_list = df["label"].to_list()
    data = []
    for vidx, video_path in enumerate(tqdm(video_path_list)):
        video_path = video_path_list[vidx]
        label = label_list[vidx]
        box_dict = get_avi_container(video_path)
        result = []
        for idx, box in enumerate(box_dict):
            help_fun_v4("", box, result, idx+1)
        result = [s.replace(":", "/").replace("&", "/").replace("=", "/").replace(" ", "") for s in result]
        result = [s[:-1] if s[-1] == "/" else s for s in result ]
        data.append({
            "filename": video_path,
            "label": label,
            "data": result
        })
    f = open(save_path, "w", encoding="utf-8")
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save CSV data to JSON file.")
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to the input CSV file.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save the output JSON file.'
    )

    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        save_path=args.save_path
    )

    

