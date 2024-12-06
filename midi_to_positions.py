#########
#PARSING#
#########
import mido
from typing import Union
KEY_SPACING_MM = 23.5  # Width of a key in millimeters
MIDDLE_C_NOTE = 60     # MIDI note number for Middle C
MAX_OCTAVE_SPAN = 12   # Number of semitones in one octave
MAX_SPEED_MM_PER_SEC = 500  # Maximum allowed speed of the arm in mm/s
NOTE_OFF_HEIGHT = 50 
PRESS_NOTE_TRAVEL_DURATION = 0.5
LIFT_NOTE_TRAVEL_DURATION = 0.5

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
    
    # We'll track the time of the previous event and only add entries when time changes
    prev_time = None
    
    for event in events:
        time = event['time']
        note = event['note']
        
        # If this is the first event or the time has changed, 
        # create a new timeline entry from the current state of active_notes.
        if time != prev_time:
            # Before we move on, if prev_time is not None, push the current state into timeline
            if prev_time is not None:
                timeline.append({'time': prev_time, 'notes': active_notes.copy()})
            prev_time = time
        
        # Update active notes according to the event
        if event['type'] == 'on':
            active_notes.add(note)
        else:
            active_notes.discard(note)
    
    # After processing all events, add the final state
    if prev_time is not None:
        timeline.append({'time': prev_time, 'notes': active_notes.copy()})
    
    return timeline


def build_segments(timeline, total_length):
        # If no timeline events, no segments are generated
    if not timeline:
        return []

    segments = []
    # Iterate over consecutive pairs of timeline entries
    for i in range(len(timeline) - 1):
        start_time = timeline[i]['time']
        end_time = timeline[i + 1]['time']
        notes = timeline[i]['notes']
        
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'notes': notes
        })
    
    # Create the final segment from the last event to the end of the track
    segments.append({
        'start_time': timeline[-1]['time'],
        'end_time': total_length,
        'notes': timeline[-1]['notes']
    })

    return segments


class Position:
    time = 0
    position_mm = 0
    gripper_width_mm = 0
    height_mm = 0
    def __init__(self, time, position, width, height):
        self.time = time
        self.position_mm = position
        self.gripper_width_mm = width
        self.height_mm = height
    
def validate_and_compute_positions(segments) -> Union[list[Position], list]:
    positions = []
    prev_position = None
    prev_width = 0
    prev_time = None
    errors = []
    
    for segment in segments:
        start_time = segment['start_time']
        end_time = segment['end_time']
        notes = segment['notes']
        
        # Allow zero, one, or two notes.
        if len(notes) > 2:
            errors.append(f"Invalid number of notes ({len(notes)}) at time {start_time:.2f}s")
            continue
        
        if len(notes) == 0:
            # No notes: Keep the same horizontal position if we have one
            # If we don't have a previous position, default to 0 mm (around Middle C)
            position_mm = prev_position if prev_position is not None else 0.0
            gripper_width_mm = prev_width
            height_mm = NOTE_OFF_HEIGHT
        else:
            # One or two notes
            notes_sorted = sorted(notes)
            offsets = [note - MIDDLE_C_NOTE for note in notes_sorted]
            positions_mm = [offset * KEY_SPACING_MM for offset in offsets]
            
            if len(notes) == 1:
                position_mm = positions_mm[0]
                gripper_width_mm = 0.0
            else:
                # Condition (b): Notes within one octave
                if abs(notes_sorted[1] - notes_sorted[0]) > MAX_OCTAVE_SPAN:
                    errors.append(f"Notes exceed one octave at time {start_time:.2f}s: {notes_sorted}")
                    continue
                position_mm = sum(positions_mm) / 2
                gripper_width_mm = abs(positions_mm[1] - positions_mm[0])
                prev_width = gripper_width_mm
            
            height_mm = 0.0  # Arm at keyboard level when notes are played
        
        # Condition (c): Arm speed limit in horizontal direction
        if prev_position is not None:
            time_diff = start_time - prev_time
            if time_diff > 0:
                speed = abs(position_mm - prev_position) / time_diff
                if speed > MAX_SPEED_MM_PER_SEC:
                    errors.append(
                        f"Arm moves too fast between {prev_time:.2f}s and {start_time:.2f}s: {speed:.2f} mm/s"
                    )
                    continue
            else:
                errors.append(f"Zero time difference between {prev_time:.2f}s and {start_time:.2f}s")
                continue
        
        positions.append(Position(start_time, position_mm, gripper_width_mm, height_mm))
        
        prev_position = position_mm
        prev_time = start_time
    
    return positions, errors

class Waypoint:
    def __init__(self, position_mm, height_mm, travel_time, arrival_time):
        self.arrival_time = arrival_time
        self.position_mm = position_mm
        self.height_mm = height_mm
        self.travel_time = travel_time

    arrival_time = 0
    position_mm = 0
    height_mm = 0
    travel_time = 0

def build_trajectory_from_positions(positions: list[Position]):
    waypoints: list[Waypoint] = []
    if not positions:
        return waypoints

    # Helper function to append a waypoint, automatically computing travel_time
    # based on the previous waypoint's arrival_time.
    def add_waypoint(position_mm, height_mm, arrival_time):
        if waypoints:
            travel_time = arrival_time - waypoints[-1].arrival_time
        else:
            travel_time = 0
        waypoints.append(Waypoint(position_mm, height_mm, travel_time, arrival_time))

    # Start from the first position
    first_pos = positions[0]

    if first_pos.height_mm == 0:
        # If the first position is a pressed note, start above the key before pressing
        start_time = first_pos.time - PRESS_NOTE_TRAVEL_DURATION
        # Add starting waypoint above the note
        add_waypoint(first_pos.position_mm, NOTE_OFF_HEIGHT, start_time)
        # Now at the note time, press down
        add_waypoint(first_pos.position_mm, 0.0, first_pos.time)
    else:
        # If the first position is not pressing a note, just start at that position/time
        # Start "from above"
        add_waypoint(first_pos.position_mm, NOTE_OFF_HEIGHT, first_pos.time)

    # Process subsequent positions
    for i in range(1, len(positions)):
        curr_pos = positions[i]
        prev_pos = positions[i-1]

        if prev_pos.height_mm == 0 and curr_pos.height_mm != 0:
            # The previous position was a pressed note. Lift up after playing the note
            lift_time = prev_pos.time + LIFT_NOTE_TRAVEL_DURATION
            add_waypoint(prev_pos.position_mm, NOTE_OFF_HEIGHT, lift_time)

        # Move horizontally to the new position
        # Arrival time is the current position's time minus PRESS_NOTE_TRAVEL_DURATION if we need to press
        # the note, or exactly at the position time if we do not press.
        if curr_pos.height_mm == 0:
            # We must arrive above the key before pressing it
            approach_time = curr_pos.time - PRESS_NOTE_TRAVEL_DURATION
            add_waypoint(curr_pos.position_mm, NOTE_OFF_HEIGHT, approach_time)
            # Press down at the note time
            add_waypoint(curr_pos.position_mm, 0.0, curr_pos.time)
        else:
            # Just arrive at the position at NOTE_OFF_HEIGHT at the given time
            #add_waypoint(curr_pos.position_mm, NOTE_OFF_HEIGHT, curr_pos.time)
            pass

    # After the final position, if it was a pressed note, lift up
    if positions[-1].height_mm == 0:
        final_lift_time = positions[-1].time + LIFT_NOTE_TRAVEL_DURATION
        add_waypoint(positions[-1].position_mm, NOTE_OFF_HEIGHT, final_lift_time)

    return waypoints


####
#IK#
####

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


L1 = 0.03923
L2 = 0.20399
L3 = 0.236
L4 = 0.14348
L2_ANGULAR_OFFSET = 0.10472
key_axis_rotation = 0


def inverse_kinematics(x, y, z, L1, L2, L3, elbow_up=False):
    q1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    Xp = r
    Zp = z - L1

    D = np.sqrt(Xp**2 + Zp**2)
    if D > (L2 + L3) or D < abs(L2 - L3):
        raise ValueError("Target not reachable")

    cos_q3 = (L2**2 + L3**2 - D**2)/(2*L2*L3)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)

    # Choose elbow configuration
    if elbow_up:
        q3 = np.pi + np.arccos(cos_q3)
    else:
        q3 = np.pi - np.arccos(cos_q3)

    alpha = np.arctan2(Zp, Xp)
    beta = np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))
    q2 = alpha - beta

    return q1, q2, q3


def frame_vectors(q1, q2, q3):
    dx = np.cos(q1)
    dy = np.sin(q1)
    x_prime = np.array([dx, dy, 0.0])
    y_prime = np.array([-dy, dx, 0.0])
    z_prime = np.array([0.0, 0.0, 1.0])

    theta = q2 + q3
    x_double_prime = x_prime * np.cos(theta) + z_prime * np.sin(theta)
    z_double_prime = -x_prime * np.sin(theta) + z_prime * np.cos(theta)

    return x_double_prime, y_prime, z_double_prime

def compute_q4_q5(q1, q2, q3):
    x_final, y_final, z_final = frame_vectors(q1, q2, q3)

    # Compute q4 to point tool straight down (-Z)
    xz = x_final[2]   # z-component of x_final
    zz = z_final[2]   # z-component of z_final
    # q4 = atan2(-zz, -xz)
    q4 = np.arctan2(-zz, -xz)

    # Compute q5 to align horizontal axis with global X.
    # Simple approach: q5 = -q1
    q5 = -q1+key_axis_rotation

    return q4, q5


def forward_kinematics(q1, q2, q3, q4, q5, L1, L2, L3, L4):
    O0 = np.array([0.0, 0.0, 0.0])
    O1 = np.array([0.0, 0.0, L1])  # top of the vertical link

    dx = np.cos(q1)
    dy = np.sin(q1)
    x_dir = np.array([dx, dy, 0.0])
    z_dir = np.array([0.0, 0.0, 1.0])

    # Joint 2 position
    O2 = O1 + L2 * (np.cos(q2)*x_dir + np.sin(q2)*z_dir)
    # Joint 3 position
    O3 = O2 + L3 * (np.cos(q2+q3)*x_dir + np.sin(q2+q3)*z_dir)

    # After q4, the tool points down: tool_z = [0,0,-1].
    O4 = O3 + L4 * np.array([0,0,-1])

    # q5 rotates about the tool's down axis- correcting for q1
    q1_q5 = q1+q5
    tool_x_dir = np.array([np.cos(q1_q5), np.sin(q1_q5), 0.0])

    O5 = O4 + 0.1 * tool_x_dir  # 0.1 is an arbitrary length for visualization

    return O0, O1, O2, O3, O4, O5

class JointSet:
    joint_positions = []
    waypoint : Waypoint = None

def waypoints_to_joint_sets(waypoints : list[Waypoint]):
    result = []
    for waypoint in waypoints:
        joint_set = JointSet()
        q1, q2, q3 = inverse_kinematics(waypoint.position_mm/1000, 100/1000, (waypoint.height_mm+200)/1000, L1, L2, L3, True)
        q4, q5 = compute_q4_q5(q1, q2, q3)
        joint_set.joint_positions = [q1, q2, q3, q4, q5]
        joint_set.waypoint = waypoint
        result.append(joint_set)

    return result

def cubic_trajectory(q0, qf, t0, tf, v0, vf):
    M = np.array([
        [1, t0, t0**2, t0**3],
        [0, 1, 2*t0, 3*(t0**2)],
        [1, tf, tf**2, tf**3],
        [0, 1, 2*tf, 3*(tf**2)]
    ], dtype=float)
    Q = np.array([q0, v0, qf, vf], dtype=float)
    a_vals = np.linalg.solve(M, Q)
    return a_vals

def generate_fine_grained_joint_sets(joint_sets, frequency=20):
    """
    Given a list of joint_sets (each with joint_positions and an arrival_time from the waypoint),
    generate a finer set of joint sets by interpolating between them using cubic trajectories.
    """
    if not joint_sets:
        return []

    fine_sets = []
    # Add the first joint_set as is.
    fine_sets.append(joint_sets[0])

    for i in range(len(joint_sets)-1):
        start_set = joint_sets[i]
        end_set = joint_sets[i+1]

        t_start = start_set.waypoint.arrival_time
        t_end = end_set.waypoint.arrival_time
        duration = t_end - t_start

        if duration <= 0:
            # No movement in time, just skip
            fine_sets.append(end_set)
            continue

        q_start = start_set.joint_positions
        q_end = end_set.joint_positions

        # Assume zero start/end velocities
        v0 = 0
        vf = 0

        # Compute cubic coefficients for each joint
        joint_trajs = []
        for j in range(len(q_start)):
            a = cubic_trajectory(q_start[j], q_end[j], 0, duration, v0, vf)
            joint_trajs.append(a)

        num_samples = int(duration * frequency)
        for s in range(1, num_samples + 1):
            t = (s / frequency)  # time from start of this segment
            if t > duration:
                t = duration
            qs = []
            for a in joint_trajs:
                # a = [a0, a1, a2, a3]
                a0, a1, a2, a3 = a
                q = a0 + a1*t + a2*(t**2) + a3*(t**3)
                qs.append(q)
            
            new_time = t_start + t
            new_wp = Waypoint(0,0,0,new_time)  # Dummy Waypoint, we only need arrival_time
            # Set arrival_time properly
            new_wp.arrival_time = new_time
            js = JointSet()
            js.joint_positions = qs
            js.waypoint = new_wp
            fine_sets.append(js)

        fine_sets.append(end_set)
    return fine_sets

def visualize_joint_sets(joint_sets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    arm_lines = []
    for _ in range(5):
        line, = ax.plot([], [], [], 'o-', lw=2)
        arm_lines.append(line)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('5-DOF Arm Trajectory from MIDI')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1])

    def update(frame):
        joint_set = joint_sets[frame]
        q1, q2, q3, q4, q5 = joint_set.joint_positions

        O0, O1, O2, O3, O4, O5 = forward_kinematics(q1, q2, q3, q4, q5, L1, L2, L3, L4)

        # Update the arm segments
        arm_points = [(O0, O1), (O1, O2), (O2, O3), (O3, O4), (O4, O5)]
        for i, seg in enumerate(arm_points):
            arm_lines[i].set_data([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]])
            arm_lines[i].set_3d_properties([seg[0][2], seg[1][2]])

        return arm_lines

    anim = FuncAnimation(fig, update, frames=len(joint_sets), interval=50, blit=False)
    plt.show()

def main(file_path):
    mid = load_midi_file(file_path)
    events = extract_note_events(mid)
    timeline = build_note_timeline(events)
    segments = build_segments(timeline, mid.length)
    positions, errors = validate_and_compute_positions(segments)
    waypoints = build_trajectory_from_positions(positions)
    joint_sets = waypoints_to_joint_sets(waypoints)
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
        print(f"Time: {pos.time:.2f}s, Position: {pos.position_mm:.2f} mm X, {pos.height_mm:.2f} mm Z Gripper Width: {pos.gripper_width_mm:.2f} mm")

    fine_grained_joint_sets = generate_fine_grained_joint_sets(joint_sets, frequency=20)

    visualize_joint_sets(fine_grained_joint_sets)

if __name__ == "__main__":
    main('valid_midi.mid')