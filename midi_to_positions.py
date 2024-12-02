import mido

KEY_SPACING_MM = 23.5  # Width of a key in millimeters
MIDDLE_C_NOTE = 60     # MIDI note number for Middle C
MAX_OCTAVE_SPAN = 12   # Number of semitones in one octave
MAX_SPEED_MM_PER_SEC = 500  # Maximum allowed speed of the arm in mm/s

def load_midi_file(file_path):
    """Load the MIDI file."""
    return mido.MidiFile(file_path)

def extract_note_events(mid):
    events = []
    current_time = 0
    for msg in mid:
        current_time += msg.time
        if msg.type in ['note_on', 'note_off']:
            note = msg.note
            velocity = msg.velocity
            if msg.type == 'note_on' and velocity > 0:
                events.append({'time': current_time, 'note': note, 'type': 'on'})
            else:
                events.append({'time': current_time, 'note': note, 'type': 'off'})
    return events

def build_note_timeline(events):
    timeline = []
    active_notes = set()
    for event in events:
        time = event['time']
        note = event['note']
        if event['type'] == 'on':
            active_notes.add(note)
        else:
            active_notes.discard(note)
        timeline.append({'time': time, 'notes': active_notes.copy()})
    return timeline

def build_segments(timeline, total_length):
    segments = []
    prev_time = 0
    prev_notes = set()
    for entry in timeline:
        current_time = entry['time']
        current_notes = entry['notes']
        if prev_notes:
            segments.append({
                'start_time': prev_time,
                'end_time': current_time,
                'notes': prev_notes.copy()
            })
        prev_time = current_time
        prev_notes = current_notes
    if prev_notes:
        segments.append({
            'start_time': prev_time,
            'end_time': total_length,
            'notes': prev_notes.copy()
        })
    return segments

def validate_and_compute_positions(segments):
    positions = []
    prev_position = None
    prev_time = None
    errors = []
    
    for segment in segments:
        start_time = segment['start_time']
        end_time = segment['end_time']
        notes = segment['notes']
        
        # Condition (a): Only one or two notes held
        if len(notes) == 0 or len(notes) > 2:
            errors.append(f"Invalid number of notes ({len(notes)}) at time {start_time:.2f}s")
            continue
        
        # Compute position and gripper width
        notes_sorted = sorted(notes)
        offsets = [note - MIDDLE_C_NOTE for note in notes_sorted]
        positions_mm = [offset * KEY_SPACING_MM for offset in offsets]
        
        if len(notes) == 1:
            position_mm = positions_mm[0]
            gripper_width_mm = 0  # Gripper closed
        else:
            # Condition (b): Notes within one octave
            if abs(notes_sorted[1] - notes_sorted[0]) > MAX_OCTAVE_SPAN:
                errors.append(f"Notes exceed one octave at time {start_time:.2f}s: {notes_sorted}")
                continue
            position_mm = sum(positions_mm) / 2
            gripper_width_mm = abs(positions_mm[1] - positions_mm[0])
        
        # Condition (c): Arm speed limit
        if prev_position is not None:
            time_diff = start_time - prev_time
            if time_diff > 0:
                speed = abs(position_mm - prev_position) / time_diff
                if speed > MAX_SPEED_MM_PER_SEC:
                    errors.append(f"Arm moves too fast between {prev_time:.2f}s and {start_time:.2f}s: {speed:.2f} mm/s")
                    continue
            else:
                errors.append(f"Zero time difference between {prev_time:.2f}s and {start_time:.2f}s")
                continue
        
        positions.append({
            'time': start_time,
            'position_mm': position_mm,
            'gripper_width_mm': gripper_width_mm
        })
        prev_position = position_mm
        prev_time = start_time
    
    return positions, errors

def main(file_path):
    mid = load_midi_file(file_path)
    events = extract_note_events(mid)
    timeline = build_note_timeline(events)
    segments = build_segments(timeline, mid.length)
    positions, errors = validate_and_compute_positions(segments)
    
    # Output errors
    if errors:
        print("Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("All conditions satisfied.")
    
    # Output computed positions
    print("\nComputed Positions and Gripper Widths:")
    for pos in positions:
        print(f"Time: {pos['time']:.2f}s, Position: {pos['position_mm']:.2f} mm, Gripper Width: {pos['gripper_width_mm']:.2f} mm")

if __name__ == "__main__":
    main('valid_midi.mid')