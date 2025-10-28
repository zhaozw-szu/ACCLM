#!/usr/bin/env python3
"""
MTS/TS Container Parser
Parses MPEG-2 Transport Stream files and creates detailed semantic tree
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



class MTSParser:
    """MPEG-2 Transport Stream parser"""
    
    # TS Packet size
    TS_PACKET_SIZE = 188
    M2TS_PACKET_SIZE = 192  # 4-byte timecode + 188-byte TS packet
    TS_SYNC_BYTE = 0x47
    
    # Stream types
    STREAM_TYPES = {
        0x00: 'Reserved',
        0x01: 'MPEG-1 Video',
        0x02: 'MPEG-2 Video',
        0x03: 'MPEG-1 Audio',
        0x04: 'MPEG-2 Audio',
        0x05: 'Private Section',
        0x06: 'PES Private Data',
        0x0F: 'MPEG-2 AAC Audio',
        0x10: 'MPEG-4 Video',
        0x11: 'MPEG-4 AAC Audio (LATM)',
        0x1B: 'H.264/AVC Video',
        0x1C: 'MPEG-4 Audio',
        0x24: 'H.265/HEVC Video',
        0x81: 'AC-3 Audio',
        0x82: 'DTS Audio',
        0x83: 'TrueHD Audio',
        0x84: 'AC-3 Plus Audio',
        0x85: 'DTS-HD Audio',
        0x86: 'DTS-HD Master Audio',
    }
    
    # Table IDs
    TABLE_IDS = {
        0x00: 'PAT (Program Association Table)',
        0x01: 'CAT (Conditional Access Table)',
        0x02: 'PMT (Program Map Table)',
        0x03: 'TSDT (Transport Stream Description Table)',
    }
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.file = None
        
        # Parsed data
        self.programs = {}
        self.pids = {}
        self.packets_analyzed = 0
        self.max_packets = 50000  # Increased limit for more detail
        self.packet_size = self.TS_PACKET_SIZE
        self.is_m2ts = False
        self.pes_packets = []  # Store PES packet info
        
    def parse(self) -> Dict[str, Any]:
        """Parse the MTS file and return semantic tree"""
        with open(self.filepath, 'rb') as f:
            self.file = f
            
            # Detect M2TS format (4-byte timestamp header)
            first_bytes = f.read(20)
            f.seek(0)
            
            # Check if it's M2TS (sync byte at position 4)
            if len(first_bytes) >= 5 and first_bytes[4] == self.TS_SYNC_BYTE:
                self.is_m2ts = True
                self.packet_size = self.M2TS_PACKET_SIZE
            
            # Parse packets
            packets_sample = []
            while f.tell() < self.file_size and self.packets_analyzed < self.max_packets:
                packet_offset = f.tell()
                packet = self._parse_packet(packet_offset)
                
                if packet:
                    # Store first 50 packets for inspection
                    if len(packets_sample) < 50:
                        packets_sample.append(packet)
                    
                    self.packets_analyzed += 1
                else:
                    break
            
            # Build result as flat array like MP4Box.js
            result = []
            
            # Add file info as first element
            file_info = {
                'type': 'ts_file',
                'format': 'MPEG-2 Transport Stream (M2TS)' if self.is_m2ts else 'MPEG-2 Transport Stream',
                'file_size': self.file_size,
                'packet_size': self.packet_size,
                'total_packets': self.file_size // self.packet_size,
                'packets_analyzed': self.packets_analyzed,
                'is_m2ts': self.is_m2ts
            }
            
            # Device detection hints from M2TS format
            if self.is_m2ts:
                file_info['container_hint'] = 'AVCHD/Blu-ray'
                file_info['typical_devices'] = ['Camcorders', 'Blu-ray recorders', 'AVCHD cameras']
            
            # Analyze packet characteristics
            if self.packets_analyzed > 0:
                # Calculate stream distribution
                total_stream_packets = sum(self.pids[pid].get('count', 0) for pid in self.pids if pid not in [0, 0x1FFF])
                if total_stream_packets > 0:
                    file_info['stream_packet_distribution'] = {
                        f'0x{pid:04X}': {
                            'count': self.pids[pid].get('count', 0),
                            'percentage': round(self.pids[pid].get('count', 0) * 100.0 / total_stream_packets, 2)
                        }
                        for pid in sorted(self.pids.keys()) 
                        if pid not in [0, 0x1FFF] and self.pids[pid].get('count', 0) > 0
                    }
            
            result.append(file_info)
            
            # Add programs
            for program_num, program_info in self.programs.items():
                program = {
                    'type': 'program',
                    'program_number': program_num,
                    'pmt_pid': program_info.get('pmt_pid'),
                    'pcr_pid': program_info.get('pcr_pid'),
                    'streams': program_info.get('streams', [])
                }
                result.append(program)
            
            # Add stream statistics
            for pid, pid_info in self.pids.items():
                if pid_info.get('stream_type') is not None:
                    stream = {
                        'type': 'stream',
                        'pid': pid,
                        'pid_hex': f'0x{pid:04X}',
                        'stream_type': pid_info['stream_type'],
                        'stream_type_name': self.STREAM_TYPES.get(pid_info['stream_type'], f'Unknown (0x{pid_info["stream_type"]:02X})'),
                        'packet_count': pid_info.get('packet_count', 0)
                    }
                    
                    # Codec analysis for device hints
                    stream_type = pid_info['stream_type']
                    if stream_type == 0x1B:  # H.264/AVC
                        stream['codec_hint'] = 'Modern HD camcorder or DSLR'
                    elif stream_type == 0x24:  # HEVC
                        stream['codec_hint'] = '4K/8K capable device'
                    elif stream_type == 0x81:  # AC-3
                        stream['audio_hint'] = 'Professional or consumer HD device'
                    elif stream_type == 0x06:  # Private data
                        stream['data_hint'] = 'Possible subtitles or auxiliary data'
                    
                    # Add descriptors if available
                    es_info = pid_info.get('es_info', {})
                    if es_info.get('descriptors'):
                        stream['descriptors'] = es_info['descriptors']
                    
                    result.append(stream)
            
            # Add PES packets (like samples in MP4)
            for pes in self.pes_packets:
                result.append(pes)
            
            # Add packet samples at the end
            if packets_sample:
                packets_container = {
                    'type': 'ts_packets_sample',
                    'count': len(packets_sample),
                    'note': 'First 50 TS packets for inspection',
                    'packets': packets_sample
                }
                result.append(packets_container)
            
            self.file = None
            return result
    
    def _parse_packet(self, offset: int) -> Optional[Dict[str, Any]]:
        """Parse a single TS packet"""
        data = self.file.read(self.packet_size)
        if len(data) < self.packet_size:
            return None
        
        # Handle M2TS timestamp
        ts_offset = 0
        if self.is_m2ts:
            # Read 4-byte timestamp
            timestamp = struct.unpack('>I', data[0:4])[0]
            ts_offset = 4
        
        # Check sync byte
        sync_byte = data[ts_offset]
        if sync_byte != self.TS_SYNC_BYTE:
            # Try to find next sync byte
            return None
        
        # Parse header
        header_bytes = struct.unpack('>I', data[ts_offset:ts_offset+4])[0]
        
        sync = (header_bytes >> 24) & 0xFF
        tei = (header_bytes >> 23) & 0x01  # Transport Error Indicator
        pusi = (header_bytes >> 22) & 0x01  # Payload Unit Start Indicator
        tp = (header_bytes >> 21) & 0x01  # Transport Priority
        pid = (header_bytes >> 8) & 0x1FFF
        tsc = (header_bytes >> 6) & 0x03  # Transport Scrambling Control
        afc = (header_bytes >> 4) & 0x03  # Adaptation Field Control
        cc = header_bytes & 0x0F  # Continuity Counter
        
        packet = {
            'offset': offset,
            'sync_byte': sync,
            'transport_error_indicator': tei,
            'payload_unit_start_indicator': pusi,
            'transport_priority': tp,
            'pid': pid,
            'transport_scrambling_control': tsc,
            'adaptation_field_control': afc,
            'continuity_counter': cc
        }
        
        if self.is_m2ts:
            packet['m2ts_timestamp'] = timestamp
        
        # Track PID
        if pid not in self.pids:
            self.pids[pid] = {
                'pid': pid,
                'packet_count': 0,
                'stream_type': None
            }
        self.pids[pid]['packet_count'] += 1
        
        # Parse adaptation field if present
        pos = ts_offset + 4
        if afc == 0x02 or afc == 0x03:
            af_length = data[pos]
            packet['adaptation_field_length'] = af_length
            
            if af_length > 0:
                af_flags = data[pos + 1]
                packet['adaptation_field'] = {
                    'discontinuity_indicator': (af_flags >> 7) & 0x01,
                    'random_access_indicator': (af_flags >> 6) & 0x01,
                    'es_priority_indicator': (af_flags >> 5) & 0x01,
                    'pcr_flag': (af_flags >> 4) & 0x01,
                    'opcr_flag': (af_flags >> 3) & 0x01,
                    'splicing_point_flag': (af_flags >> 2) & 0x01,
                    'transport_private_data_flag': (af_flags >> 1) & 0x01,
                    'adaptation_field_extension_flag': af_flags & 0x01
                }
                
                # Parse PCR if present
                if (af_flags >> 4) & 0x01 and af_length >= 7:
                    pcr_bytes = data[pos + 2:pos + 8]
                    pcr_base = (struct.unpack('>Q', b'\x00\x00' + pcr_bytes)[0] >> 15) & 0x1FFFFFFFF
                    pcr_ext = struct.unpack('>H', pcr_bytes[4:6])[0] & 0x1FF
                    packet['adaptation_field']['pcr'] = pcr_base * 300 + pcr_ext
            
            pos += 1 + af_length
        
        # Parse payload
        if afc == 0x01 or afc == 0x03:
            payload = data[pos:]
            
            # Parse specific PIDs
            if pid == 0x0000:  # PAT
                if pusi:
                    self._parse_pat(payload)
            elif pid in [p.get('pmt_pid') for p in self.programs.values()]:  # PMT
                if pusi:
                    self._parse_pmt(payload, pid)
            elif pusi and pid in self.pids and self.pids[pid].get('stream_type') is not None:
                # Parse PES packet start
                if len(self.pes_packets) < 1000:  # Limit PES packet storage
                    pes_info = self._parse_pes_header(payload, pid, offset)
                    if pes_info:
                        self.pes_packets.append(pes_info)
        
        return packet
    
    def _parse_pat(self, payload: bytes):
        """Parse Program Association Table"""
        if len(payload) < 8:
            return
        
        # Skip pointer field if present
        pointer = payload[0]
        pos = 1 + pointer
        
        if pos >= len(payload):
            return
        
        # Parse table header
        table_id = payload[pos]
        section_length = ((payload[pos + 1] & 0x0F) << 8) | payload[pos + 2]
        transport_stream_id = (payload[pos + 3] << 8) | payload[pos + 4]
        
        pos += 8  # Skip to program data
        
        # Parse programs
        num_programs = (section_length - 9) // 4
        for i in range(num_programs):
            if pos + 4 > len(payload):
                break
            
            program_num = (payload[pos] << 8) | payload[pos + 1]
            pmt_pid = ((payload[pos + 2] & 0x1F) << 8) | payload[pos + 3]
            
            if program_num != 0:  # Ignore NIT
                self.programs[program_num] = {
                    'program_number': program_num,
                    'pmt_pid': pmt_pid,
                    'streams': []
                }
            
            pos += 4
    
    def _parse_pmt(self, payload: bytes, pmt_pid: int):
        """Parse Program Map Table"""
        if len(payload) < 12:
            return
        
        # Skip pointer field
        pointer = payload[0]
        pos = 1 + pointer
        
        if pos >= len(payload):
            return
        
        # Parse table header
        table_id = payload[pos]
        section_length = ((payload[pos + 1] & 0x0F) << 8) | payload[pos + 2]
        program_number = (payload[pos + 3] << 8) | payload[pos + 4]
        
        pcr_pid = ((payload[pos + 8] & 0x1F) << 8) | payload[pos + 9]
        program_info_length = ((payload[pos + 10] & 0x0F) << 8) | payload[pos + 11]
        
        pos += 12 + program_info_length
        
        # Find program
        program = None
        for prog in self.programs.values():
            if prog['pmt_pid'] == pmt_pid:
                program = prog
                program['pcr_pid'] = pcr_pid
                break
        
        if not program:
            return
        
        # Parse elementary streams
        while pos + 5 <= len(payload) - 4:  # -4 for CRC
            stream_type = payload[pos]
            elementary_pid = ((payload[pos + 1] & 0x1F) << 8) | payload[pos + 2]
            es_info_length = ((payload[pos + 3] & 0x0F) << 8) | payload[pos + 4]
            
            stream_info = {
                'stream_type': stream_type,
                'stream_type_name': self.STREAM_TYPES.get(stream_type, f'Unknown (0x{stream_type:02X})'),
                'elementary_pid': elementary_pid,
                'es_info_length': es_info_length
            }
            
            # Parse descriptors
            if es_info_length > 0 and pos + 5 + es_info_length <= len(payload):
                descriptors = self._parse_descriptors(payload[pos + 5:pos + 5 + es_info_length])
                if descriptors:
                    stream_info['descriptors'] = descriptors
                    
                    # Check for HDMV (Blu-ray) registration
                    for desc in descriptors:
                        if desc.get('tag') == 5:
                            format_id = desc.get('format_identifier', '')
                            if format_id == 'HDMV':
                                stream_info['container_hint'] = 'AVCHD/Blu-ray'
                                stream_info['device_category'] = 'Consumer camcorder or Blu-ray recorder'
                                stream_info['device_examples'] = ['Sony Handycam', 'Panasonic camcorder', 'Canon VIXIA']
                            elif format_id == 'AC-3':
                                stream_info['audio_format'] = 'Dolby Digital'
                                stream_info['quality_hint'] = 'Professional or high-end consumer'
            
            program['streams'].append(stream_info)
            
            # Update PID info
            if elementary_pid in self.pids:
                self.pids[elementary_pid]['stream_type'] = stream_type
                self.pids[elementary_pid]['es_info'] = stream_info
            
            pos += 5 + es_info_length
    
    def _parse_descriptors(self, data: bytes) -> List[Dict[str, Any]]:
        """Parse descriptor data"""
        descriptors = []
        pos = 0
        
        while pos + 2 <= len(data):
            tag = data[pos]
            length = data[pos + 1]
            
            if pos + 2 + length > len(data):
                break
            
            descriptor = {
                'tag': tag,
                'tag_name': self._get_descriptor_name(tag),
                'length': length
            }
            
            # Parse specific descriptors
            if tag == 0x0A and length >= 4:  # ISO 639 language descriptor
                lang_code = data[pos + 2:pos + 5].decode('ascii', errors='ignore')
                audio_type = data[pos + 5] if pos + 5 < pos + 2 + length else 0
                descriptor['language_code'] = lang_code
                descriptor['audio_type'] = audio_type
            elif tag == 0x05 and length >= 4:  # Registration descriptor
                format_id = data[pos + 2:pos + 6].decode('ascii', errors='ignore')
                descriptor['format_identifier'] = format_id
            elif length <= 64:  # Store small descriptor data
                descriptor['data'] = list(data[pos + 2:pos + 2 + length])
            
            descriptors.append(descriptor)
            pos += 2 + length
        
        return descriptors
    
    def _get_descriptor_name(self, tag: int) -> str:
        """Get descriptor name from tag"""
        descriptors = {
            0x02: 'video_stream_descriptor',
            0x03: 'audio_stream_descriptor',
            0x05: 'registration_descriptor',
            0x06: 'data_stream_alignment_descriptor',
            0x09: 'CA_descriptor',
            0x0A: 'ISO_639_language_descriptor',
            0x0E: 'maximum_bitrate_descriptor',
            0x1B: 'MPEG-4_video_descriptor',
            0x1C: 'MPEG-4_audio_descriptor',
            0x28: 'AVC_video_descriptor',
            0x38: 'HEVC_video_descriptor',
        }
        return descriptors.get(tag, f'descriptor_0x{tag:02X}')
    
    def _parse_pes_header(self, payload: bytes, pid: int, ts_offset: int) -> Optional[Dict[str, Any]]:
        """Parse PES packet header"""
        if len(payload) < 9:
            return None
        
        # Skip pointer field if present
        if payload[0] != 0x00:
            return None
        
        # Check PES start code
        if payload[0:3] != b'\x00\x00\x01':
            return None
        
        stream_id = payload[3]
        pes_packet_length = (payload[4] << 8) | payload[5]
        
        pes_info = {
            'type': 'pes_packet',
            'pid': pid,
            'pid_hex': f'0x{pid:04X}',
            'ts_packet_offset': ts_offset,
            'stream_id': stream_id,
            'stream_id_hex': f'0x{stream_id:02X}',
            'packet_length': pes_packet_length
        }
        
        # Parse PES header for streams with timestamps
        if stream_id not in [0xBC, 0xBE, 0xBF, 0xF0, 0xF1, 0xF2, 0xF8, 0xFF]:
            if len(payload) >= 9:
                pes_flags = payload[7]
                pts_dts_flags = (pes_flags >> 6) & 0x03
                
                pes_info['has_pts'] = bool(pts_dts_flags & 0x02)
                pes_info['has_dts'] = bool(pts_dts_flags & 0x01)
                
                header_data_length = payload[8]
                pes_info['header_length'] = header_data_length
                
                # Parse PTS
                if pts_dts_flags & 0x02 and len(payload) >= 14:
                    pts = ((payload[9] & 0x0E) << 29) | \
                          ((payload[10] & 0xFF) << 22) | \
                          ((payload[11] & 0xFE) << 14) | \
                          ((payload[12] & 0xFF) << 7) | \
                          ((payload[13] & 0xFE) >> 1)
                    pes_info['pts'] = pts
                    pes_info['pts_seconds'] = pts / 90000.0
                
                # Parse DTS
                if pts_dts_flags & 0x01 and len(payload) >= 19:
                    dts = ((payload[14] & 0x0E) << 29) | \
                          ((payload[15] & 0xFF) << 22) | \
                          ((payload[16] & 0xFE) << 14) | \
                          ((payload[17] & 0xFF) << 7) | \
                          ((payload[18] & 0xFE) >> 1)
                    pes_info['dts'] = dts
                    pes_info['dts_seconds'] = dts / 90000.0
        
        return pes_info


def format_bytes(bytes_value: int) -> str:
    """Format byte size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def get_mts_container(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    print(f"Parsing MTS file: {input_file}")
    print(f"File size: {format_bytes(os.path.getsize(input_file))}")
    print()
    
    # Parse the file
    parser = MTSParser(input_file)
    tree = parser.parse()
    
    # Print summary
    print("=== MTS File Summary ===")
    
    # Extract info from flat structure
    file_info = next((item for item in tree if item.get('type') == 'ts_file'), {})
    programs = [item for item in tree if item.get('type') == 'program']
    streams = [item for item in tree if item.get('type') == 'stream']
    
    print(f"Format: {file_info.get('format', 'Unknown')}")
    print(f"Total packets: {file_info.get('total_packets', 0)}")
    print(f"Packets analyzed: {file_info.get('packets_analyzed', 0)}")
    print(f"Programs found: {len(programs)}")
    print(f"Streams found: {len(streams)}")
    print()
    
    for prog in programs:
        print(f"Program {prog['program_number']}:")
        print(f"  PMT PID: 0x{prog['pmt_pid']:04X}")
        if prog.get('pcr_pid'):
            print(f"  PCR PID: 0x{prog['pcr_pid']:04X}")
        print(f"  Streams: {len(prog.get('streams', []))}")
        for stream in prog.get('streams', []):
            print(f"    - PID 0x{stream['elementary_pid']:04X}: {stream['stream_type_name']}")
    
    print()
    print("Stream Statistics:")
    for stream in streams:
        print(f"  PID {stream['pid_hex']}: {stream['stream_type_name']} ({stream['packet_count']} packets)")


    return tree
    


def main(csv_path, save_path):
    df = pd.read_csv(csv_path)
    video_path_list = df["filename"].to_list()
    label_list = df["label"].to_list()
    data = []
    for vidx, video_path in enumerate(tqdm(video_path_list)):
        video_path = video_path_list[vidx]
        label = label_list[vidx]
        box_dict = get_mts_container(video_path)
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

