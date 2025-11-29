"""
Motion Capture Report Generator
Generates professional pitching biomechanics reports similar to WVU Baseball Mocap Reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge
from io import BytesIO
import os

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.lib import colors


# ============================================================================
# COLOR SCHEME (WVU-style colors)
# ============================================================================
COLORS = {
    'navy': HexColor('#002855'),      # Dark navy blue (header background)
    'gold': HexColor('#EAAA00'),      # Gold (accents, titles)
    'light_gold': HexColor('#F5D78E'),
    'white': white,
    'black': black,
    'green': HexColor('#90EE90'),     # Light green for "in range"
    'yellow': HexColor('#FFFACD'),    # Light yellow for "near range"
    'red': HexColor('#FFB6C1'),       # Light red for "out of range"
    'gray': HexColor('#F5F5F5'),
    'dark_gray': HexColor('#333333'),
}


# ============================================================================
# REFERENCE DATA (MLB averages from Kinatrax - sample ranges)
# ============================================================================
REFERENCE_RANGES = {
    # Shoulder metrics at FP, MER, REL
    'shoulder_horizontal_abduction': {'FP': (-36, -12), 'MER': (-7, 15), 'REL': (-1, 23)},
    'shoulder_abduction': {'FP': (80, 100), 'MER': (82, 98), 'REL': (82, 98)},
    'shoulder_external_rotation': {'FP': (14, 90), 'MER': (170, 188), 'REL': (97, 121)},
    
    # Elbow metrics
    'elbow_flexion': {'FP': (94, 124), 'MER': (74, 94), 'REL': (20, 30)},
    
    # Trunk metrics
    'trunk_forward_tilt': {'FP': (9, 21), 'MER': (10, 22), 'REL': (26, 40)},
    'trunk_lateral_tilt': {'FP': (-10, 0), 'MER': (13, 29), 'REL': (8, 28)},
    'trunk_rotation': {'FP': (95, 118), 'MER': (-5, 15), 'REL': (-4, -24)},
    
    # Pelvis metrics
    'pelvic_forward_tilt': {'FP': (0, 14), 'MER': (32, 46), 'REL': (40, 56)},
    'pelvic_lateral_tilt': {'FP': (-3, 5), 'MER': (-3, 13), 'REL': (-5, 11)},
    'pelvic_rotation': {'FP': (30, 76), 'MER': (-10, 6), 'REL': (-14, 2)},
    
    # Hip-shoulder separation
    'hip_shoulder_separation': {'FP': (32, 52)},
    
    # Knee flexion
    'knee_flexion': {'FP': (43, 57), 'MER': (40, 60), 'REL': (28, 56)},
    
    # Stress metrics
    'shoulder_force_bw': (176, 224),
    'shoulder_internal_rotation_torque_bwh': (17, 29),
    'elbow_torque_bwh': (13, 17),
    
    # Angular velocities
    'pelvis_angular_velocity': (581, 811),
    'trunk_angular_velocity': (861, 1187),
    'elbow_angular_velocity': (2030, 2542),
    'shoulder_angular_velocity': (3801, 4923),
}


def get_range_color(value, ref_range, std=None):
    """
    Determine color based on whether value is within reference range.
    Green = within range
    Yellow = within 1 std dev outside range
    Red = more than 1 std dev outside range
    """
    if ref_range is None:
        return COLORS['gray']
    
    low, high = ref_range
    if low <= value <= high:
        return COLORS['green']
    
    # Calculate how far outside the range
    if std is not None:
        if value < low:
            distance = low - value
        else:
            distance = value - high
        
        if distance <= std:
            return COLORS['yellow']
    
    return COLORS['red']


def create_time_series_plot(times, values, std_values=None, ref_range=None,
                           fp_time=0.5, mer_time=0.7, rel_time=0.85,
                           ylabel='Angle (°)', title='', figsize=(5, 3)):
    """
    Create a time series plot similar to the mocap report graphs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot reference range as shaded area
    if ref_range is not None:
        ax.fill_between(times, ref_range[0], ref_range[1], 
                       alpha=0.3, color='lightblue', label='Reference Range')
    
    # Plot mean line
    ax.plot(times, values, 'b-', linewidth=2, label='Mean')
    
    # Plot std dev shaded area if provided
    if std_values is not None:
        ax.fill_between(times, values - std_values, values + std_values,
                       alpha=0.4, color='navy')
    
    # Add vertical lines for key events
    ax.axvline(x=fp_time, color='blue', linestyle='-', linewidth=2, alpha=0.7)
    ax.axvline(x=rel_time, color='red', linestyle='-', linewidth=2, alpha=0.7)
    
    # Add marker for MER
    ax.axvline(x=mer_time, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


def create_kinematic_sequence_plot(pelvis_data, trunk_data, elbow_data, shoulder_data,
                                   times, figsize=(8, 4)):
    """
    Create the kinematic sequence plot showing angular velocities.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(times, pelvis_data, 'r-', linewidth=2, label='Pelvis')
    ax.plot(times, trunk_data, 'g-', linewidth=2, label='Trunk')
    ax.plot(times, elbow_data, 'b-', linewidth=2, label='Elbow')
    ax.plot(times, shoulder_data, color='orange', linewidth=2, label='Shoulder')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='blue', linestyle='-', linewidth=2, alpha=0.7)
    ax.axvline(x=0.85, color='red', linestyle='-', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Angular Velocities (°/s)', fontsize=10)
    ax.set_title('Kinematic Sequence', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


def create_arm_slot_diagram(arm_slot_angle, figsize=(3, 3)):
    """
    Create the arm slot pie chart / wedge diagram.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define arm slot categories
    categories = [
        ('Overhead', 0, 40, '#003366'),
        ('High Three-Quarter', 40, 50, '#004488'),
        ('Three-Quarter', 50, 60, '#0066AA'),
        ('Low Three-Quarter', 60, 70, '#0088CC'),
        ('Sidearm', 70, 90, '#EAAA00'),
        ('Low Sidearm', 90, 110, '#CC8800'),
        ('Submarine', 110, 180, '#AA6600'),
    ]
    
    # Draw wedges
    for name, start, end, color in categories:
        wedge = Wedge(center=(0.5, 0), r=0.4, theta1=90-end, theta2=90-start,
                     facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(wedge)
    
    # Add indicator line for current arm slot
    angle_rad = np.radians(90 - arm_slot_angle)
    x_end = 0.5 + 0.42 * np.cos(angle_rad)
    y_end = 0.42 * np.sin(angle_rad)
    ax.plot([0.5, x_end], [0, y_end], 'r-', linewidth=3)
    ax.plot(x_end, y_end, 'ro', markersize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


class MocapReportGenerator:
    """
    Main class for generating motion capture PDF reports.
    """
    
    def __init__(self, player_name, date, velocity_range, output_path):
        self.player_name = player_name
        self.date = date
        self.velocity_range = velocity_range
        self.output_path = output_path
        self.page_width, self.page_height = landscape(LETTER)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Data storage
        self.metrics = {}
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=COLORS['gold'],
            alignment=TA_CENTER,
            spaceAfter=6,
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=COLORS['gold'],
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=6,
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=COLORS['dark_gray'],
            alignment=TA_LEFT,
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=COLORS['navy'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
        ))
    
    def set_metrics(self, metrics_dict):
        """
        Set the metrics data for the report.
        
        Expected structure:
        {
            'shoulder_horizontal_abduction': {'FP': (mean, std), 'MER': (mean, std), 'REL': (mean, std)},
            'shoulder_abduction': {...},
            ...
            'angular_velocities': {'knee': val, 'pelvis': val, 'trunk': val, 'elbow': val, 'shoulder': val},
            'stress': {'shoulder_force': val, 'shoulder_torque': val, 'elbow_torque': val},
            'arm_slot': (angle, std),
            ...
        }
        """
        self.metrics = metrics_dict
    
    def _create_header_table(self):
        """Create the header section with player name and info."""
        # Create header data
        header_data = [
            [
                Paragraph(f"<b>Mocap Report:</b> {self.player_name}", 
                         ParagraphStyle('Header', fontSize=24, textColor=COLORS['gold'])),
            ],
            [
                Paragraph(f"<b>Date:</b> {self.date}", 
                         ParagraphStyle('SubHeader', fontSize=12, textColor=COLORS['white'])),
                Paragraph(f"<b>Velocity:</b> {self.velocity_range}", 
                         ParagraphStyle('SubHeader', fontSize=12, textColor=COLORS['gold'])),
            ],
        ]
        
        header_table = Table(header_data, colWidths=[5*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        return header_table
    
    def _create_metrics_table(self, title, metrics_list, include_timing=False):
        """
        Create a table of metrics with color-coded cells.
        
        metrics_list: List of tuples (label, fp_val, fp_std, mer_val, mer_std, rel_val, rel_std, fp_ref, mer_ref, rel_ref)
        """
        # Header row
        header = ['Key Metrics', 'Foot Plant', 'Max External Rotation', 'Release']
        
        data = [header]
        styles = [
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        
        for i, metric in enumerate(metrics_list):
            label = metric[0]
            fp_val, fp_std = metric[1], metric[2]
            mer_val, mer_std = metric[3], metric[4]
            rel_val, rel_std = metric[5], metric[6]
            fp_ref, mer_ref, rel_ref = metric[7], metric[8], metric[9]
            
            row = [
                label,
                f"{fp_val}° ± {fp_std}°",
                f"{mer_val}° ± {mer_std}°",
                f"{rel_val}° ± {rel_std}°",
            ]
            data.append(row)
            
            row_idx = i + 1  # +1 for header row
            
            # Color cells based on reference ranges
            if fp_ref:
                bg_color = get_range_color(fp_val, fp_ref, fp_std)
                styles.append(('BACKGROUND', (1, row_idx), (1, row_idx), bg_color))
            if mer_ref:
                bg_color = get_range_color(mer_val, mer_ref, mer_std)
                styles.append(('BACKGROUND', (2, row_idx), (2, row_idx), bg_color))
            if rel_ref:
                bg_color = get_range_color(rel_val, rel_ref, rel_std)
                styles.append(('BACKGROUND', (3, row_idx), (3, row_idx), bg_color))
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle(styles))
        
        return table
    
    def _create_angular_velocity_table(self):
        """Create the angular velocities summary table."""
        velocities = self.metrics.get('angular_velocities', {})
        
        data = [
            ['Angular Velocities', ''],
            ['Knee', f"{velocities.get('knee', 'N/A')}°/s"],
            ['Pelvis', f"{velocities.get('pelvis', 'N/A')}°/s"],
            ['Trunk', f"{velocities.get('trunk', 'N/A')}°/s"],
            ['Elbow', f"{velocities.get('elbow', 'N/A')}°/s"],
            ['Shoulder', f"{velocities.get('shoulder', 'N/A')}°/s"],
        ]
        
        styles = [
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('SPAN', (0, 0), (1, 0)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]
        
        # Color code based on reference ranges
        ref_keys = ['knee', 'pelvis', 'trunk', 'elbow', 'shoulder']
        for i, key in enumerate(ref_keys):
            val = velocities.get(key, 0)
            ref = REFERENCE_RANGES.get(f'{key}_angular_velocity')
            if ref:
                bg_color = get_range_color(val, ref)
                styles.append(('BACKGROUND', (1, i+1), (1, i+1), bg_color))
        
        table = Table(data, colWidths=[1*inch, 1*inch])
        table.setStyle(TableStyle(styles))
        
        return table
    
    def _create_stress_table(self):
        """Create the stress metrics summary table."""
        stress = self.metrics.get('stress', {})
        
        data = [
            ['Stress', ''],
            ['Shoulder Force', f"{stress.get('shoulder_force', 'N/A')}%BW"],
            ['Shoulder Torque', f"{stress.get('shoulder_torque', 'N/A')}%BWH"],
            ['Elbow Torque', f"{stress.get('elbow_torque', 'N/A')}%BWH"],
        ]
        
        styles = [
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('SPAN', (0, 0), (1, 0)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]
        
        # Color code
        stress_refs = [
            ('shoulder_force', REFERENCE_RANGES.get('shoulder_force_bw')),
            ('shoulder_torque', REFERENCE_RANGES.get('shoulder_internal_rotation_torque_bwh')),
            ('elbow_torque', REFERENCE_RANGES.get('elbow_torque_bwh')),
        ]
        
        for i, (key, ref) in enumerate(stress_refs):
            val = stress.get(key, 0)
            if ref:
                bg_color = get_range_color(val, ref)
                styles.append(('BACKGROUND', (1, i+1), (1, i+1), bg_color))
        
        table = Table(data, colWidths=[1.2*inch, 1*inch])
        table.setStyle(TableStyle(styles))
        
        return table
    
    def _draw_page_background(self, canvas, doc):
        """Draw page background and header."""
        canvas.saveState()
        
        # Draw header bar
        canvas.setFillColor(COLORS['navy'])
        canvas.rect(0, self.page_height - 60, self.page_width, 60, fill=True, stroke=False)
        
        # Draw title
        canvas.setFillColor(COLORS['gold'])
        canvas.setFont('Helvetica-Bold', 20)
        canvas.drawString(30, self.page_height - 40, f"Mocap Report: {self.player_name}")
        
        # Draw velocity on right
        canvas.setFont('Helvetica-Bold', 14)
        canvas.drawRightString(self.page_width - 30, self.page_height - 40, 
                               f"Velocity: {self.velocity_range}")
        
        canvas.restoreState()
    
    def generate_summary_page(self):
        """Generate the first summary page content."""
        elements = []
        
        # Add header
        elements.append(self._create_header_table())
        elements.append(Spacer(1, 12))
        
        # SUMMARY section title
        summary_title = Paragraph(
            "<font color='#EAAA00' size='24'><b>SUMMARY</b></font>",
            ParagraphStyle('SummaryTitle', alignment=TA_CENTER)
        )
        elements.append(summary_title)
        elements.append(Spacer(1, 12))
        
        # Create main metrics table
        shoulder_metrics = [
            ('Shoulder Horizontal Abduction', -44, 1, -4, 2, 1, 1, 
             REFERENCE_RANGES['shoulder_horizontal_abduction']['FP'],
             REFERENCE_RANGES['shoulder_horizontal_abduction']['MER'],
             REFERENCE_RANGES['shoulder_horizontal_abduction']['REL']),
            ('Shoulder Abduction', 91, 2, 102, 0.6, 99, 1,
             REFERENCE_RANGES['shoulder_abduction']['FP'],
             REFERENCE_RANGES['shoulder_abduction']['MER'],
             REFERENCE_RANGES['shoulder_abduction']['REL']),
            ('Shoulder External Rotation', 48, 4, 193, 1, 123, 3,
             REFERENCE_RANGES['shoulder_external_rotation']['FP'],
             REFERENCE_RANGES['shoulder_external_rotation']['MER'],
             REFERENCE_RANGES['shoulder_external_rotation']['REL']),
            ('Elbow Flexion', 113, 1, 86, 1, 30, 1,
             REFERENCE_RANGES['elbow_flexion']['FP'],
             REFERENCE_RANGES['elbow_flexion']['MER'],
             REFERENCE_RANGES['elbow_flexion']['REL']),
            ('Trunk Forward Tilt', 13, 1, 33, 1, 48, 1,
             REFERENCE_RANGES['trunk_forward_tilt']['FP'],
             REFERENCE_RANGES['trunk_forward_tilt']['MER'],
             REFERENCE_RANGES['trunk_forward_tilt']['REL']),
            ('Trunk Rotation', 109, 2, -6, 0.8, -20, 0.7,
             REFERENCE_RANGES['trunk_rotation']['FP'],
             REFERENCE_RANGES['trunk_rotation']['MER'],
             REFERENCE_RANGES['trunk_rotation']['REL']),
            ('Hip-Shoulder Separation', 51, 2, 11, 2, 6, 3,
             REFERENCE_RANGES['hip_shoulder_separation']['FP'],
             None, None),
            ('Pelvic Forward Tilt', 7, 0.6, 45, 1, 55, 2,
             REFERENCE_RANGES['pelvic_forward_tilt']['FP'],
             REFERENCE_RANGES['pelvic_forward_tilt']['MER'],
             REFERENCE_RANGES['pelvic_forward_tilt']['REL']),
            ('Pelvic Rotation', 57, 3, -9, 2, -11, 2,
             REFERENCE_RANGES['pelvic_rotation']['FP'],
             REFERENCE_RANGES['pelvic_rotation']['MER'],
             REFERENCE_RANGES['pelvic_rotation']['REL']),
            ('Knee Flexion', 61, 2, 58, 4, 48, 3,
             REFERENCE_RANGES['knee_flexion']['FP'],
             REFERENCE_RANGES['knee_flexion']['MER'],
             REFERENCE_RANGES['knee_flexion']['REL']),
        ]
        
        metrics_table = self._create_metrics_table('Key Metrics', shoulder_metrics)
        
        # Angular velocities table
        self.metrics['angular_velocities'] = {
            'knee': -338,
            'pelvis': 594,
            'trunk': 1060,
            'elbow': 2209,
            'shoulder': 4277,
        }
        
        # Stress metrics
        self.metrics['stress'] = {
            'shoulder_force': 129,
            'shoulder_torque': 7,
            'elbow_torque': 6,
        }
        
        angular_table = self._create_angular_velocity_table()
        stress_table = self._create_stress_table()
        
        # Create side-by-side layout
        main_layout = Table(
            [[metrics_table, angular_table], [None, stress_table]],
            colWidths=[7*inch, 2.5*inch]
        )
        main_layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(main_layout)
        
        return elements
    
    def generate_shoulder_page(self):
        """Generate the shoulder metrics page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Shoulder</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Generate plots (using sample data for demonstration)
        times = np.linspace(0, 1, 100)
        
        # Shoulder Horizontal Abduction plot
        sha_values = -40 + 30*np.sin(2*np.pi*times) + 20*times
        sha_plot = create_time_series_plot(
            times, sha_values, std_values=np.ones_like(times)*5,
            ref_range=(-36, 15),
            ylabel='Angle (°)',
            title='Shoulder Horizontal Abduction'
        )
        
        # Shoulder Abduction plot
        sab_values = 60 + 40*np.sin(np.pi*times)
        sab_plot = create_time_series_plot(
            times, sab_values, std_values=np.ones_like(times)*8,
            ref_range=(80, 100),
            ylabel='Angle (°)',
            title='Shoulder Abduction'
        )
        
        # Shoulder External Rotation plot
        ser_values = 20 + 150*times + 30*np.sin(4*np.pi*times)
        ser_plot = create_time_series_plot(
            times, ser_values, std_values=np.ones_like(times)*10,
            ref_range=(14, 188),
            ylabel='Angle (°)',
            title='Shoulder External Rotation'
        )
        
        # Create images
        sha_img = Image(sha_plot, width=3.2*inch, height=2.2*inch)
        sab_img = Image(sab_plot, width=3.2*inch, height=2.2*inch)
        ser_img = Image(ser_plot, width=3.2*inch, height=2.2*inch)
        
        # Create data tables for each metric
        sha_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '-44° ± 1°', '-12° - -36°'],
            ['MER', '-4° ± 2°', '-7° - 15°'],
            ['REL', '1° ± 1°', '-1° - 23°'],
        ]
        
        sab_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '91° ± 2°', '80° - 100°'],
            ['MER', '102° ± 0.6°', '82° - 98°'],
            ['REL', '99° ± 1°', '82° - 98°'],
        ]
        
        ser_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '48° ± 4°', '14° - 90°'],
            ['MER', '193° ± 1°', '170° - 188°'],
            ['REL', '123° ± 3°', '97° - 121°'],
        ]
        
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['red']),    # FP out of range
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),  # MER in range
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),  # REL in range
        ])
        
        sha_table = Table(sha_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        sha_table.setStyle(table_style)
        
        sab_table = Table(sab_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        sab_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['yellow']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['yellow']),
        ]))
        
        ser_table = Table(ser_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        ser_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['yellow']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['yellow']),
        ]))
        
        # Layout: 3 columns of plot + table
        plots_layout = Table([
            [sha_img, sab_img, ser_img],
            [sha_table, sab_table, ser_table],
        ], colWidths=[3.3*inch, 3.3*inch, 3.3*inch])
        
        plots_layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(plots_layout)
        
        # Add timing indicator
        elements.append(Spacer(1, 12))
        timing_text = Paragraph(
            "<b>Timing:</b> On Time",
            ParagraphStyle('Timing', fontSize=12, textColor=COLORS['navy'])
        )
        elements.append(timing_text)
        
        return elements
    
    def generate_kinematic_sequence_page(self):
        """Generate the kinematic sequence page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Kinematic Sequence</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Generate kinematic sequence plot
        times = np.linspace(0, 1, 100)
        
        # Sample kinematic sequence data
        pelvis_data = 600 * np.exp(-((times - 0.5)**2) / 0.02) - 200
        trunk_data = 1100 * np.exp(-((times - 0.6)**2) / 0.02) - 300
        elbow_data = 2200 * np.exp(-((times - 0.7)**2) / 0.015) - 500
        shoulder_data = 4300 * np.exp(-((times - 0.8)**2) / 0.01) - 1000
        
        kin_plot = create_kinematic_sequence_plot(
            pelvis_data, trunk_data, elbow_data, shoulder_data, times,
            figsize=(8, 4)
        )
        
        kin_img = Image(kin_plot, width=8*inch, height=4*inch)
        elements.append(kin_img)
        elements.append(Spacer(1, 12))
        
        # Peak angular velocities table
        peak_data = [
            ['', 'Order', 'Mean ± Std Dev', 'Reference'],
            ['Pelvis', '1', '594°/s ± 20°/s', '581°/s - 811°/s'],
            ['Trunk', '2', '1060°/s ± 33°/s', '861°/s - 1187°/s'],
            ['Elbow', '3', '2209°/s ± 58°/s', '2030°/s - 2542°/s'],
            ['Shoulder', '4', '4277°/s ± 162°/s', '3801°/s - 4923°/s'],
        ]
        
        peak_table = Table(peak_data, colWidths=[1.2*inch, 0.8*inch, 1.8*inch, 1.8*inch])
        peak_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (2, 1), (2, 1), COLORS['green']),
            ('BACKGROUND', (2, 2), (2, 2), COLORS['green']),
            ('BACKGROUND', (2, 3), (2, 3), COLORS['green']),
            ('BACKGROUND', (2, 4), (2, 4), COLORS['green']),
        ]))
        
        elements.append(peak_table)
        
        return elements
    
    def generate_elbow_arm_slot_page(self):
        """Generate the elbow flexion and arm slot page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Elbow / Arm Slot</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Generate elbow flexion plot
        times = np.linspace(0, 1, 100)
        elbow_values = 120 - 50*times + 40*np.sin(2*np.pi*times)
        
        elbow_plot = create_time_series_plot(
            times, elbow_values, std_values=np.ones_like(times)*5,
            ref_range=(20, 124),
            ylabel='Elbow Flexion Angle (°)',
            title='Elbow Flexion',
            figsize=(5, 3)
        )
        
        elbow_img = Image(elbow_plot, width=4.5*inch, height=3*inch)
        
        # Elbow data table
        elbow_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '113° ± 1°', '94° - 124°'],
            ['MER', '86° ± 1°', '74° - 94°'],
            ['REL', '30° ± 1°', '20° - 30°'],
        ]
        
        elbow_table = Table(elbow_data, colWidths=[0.6*inch, 1.3*inch, 1.2*inch])
        elbow_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),
        ]))
        
        # Arm slot description and visualization
        arm_slot_plot = create_arm_slot_diagram(62, figsize=(3, 3))
        arm_slot_img = Image(arm_slot_plot, width=2.5*inch, height=2.5*inch)
        
        arm_slot_info = [
            [Paragraph("<b>Arm Slot</b>", ParagraphStyle('Header', fontSize=14, textColor=COLORS['navy']))],
            [Paragraph("Arm Slot (°): <b><font color='#0088CC'>62° ± 2°</font></b>", 
                      ParagraphStyle('Info', fontSize=12))],
            [Paragraph("Arm Slot: <b><font color='#0088CC'>Low Three-Quarter</font></b>", 
                      ParagraphStyle('Info', fontSize=12))],
        ]
        
        arm_slot_table = Table(arm_slot_info, colWidths=[3*inch])
        arm_slot_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        # Layout
        layout = Table([
            [elbow_img, arm_slot_img],
            [elbow_table, arm_slot_table],
        ], colWidths=[5*inch, 4.5*inch])
        
        layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(layout)
        
        return elements
    
    def generate_stress_page(self):
        """Generate the throwing arm stress page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Throwing Arm Stress</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        times = np.linspace(0, 1, 100)
        
        # Shoulder Force plot
        force_values = 200 + 1200*np.exp(-((times - 0.7)**2) / 0.02)
        force_plot = create_time_series_plot(
            times, force_values, std_values=np.ones_like(times)*50,
            ref_range=None,
            ylabel='Shoulder Distraction Force (N)',
            title='Shoulder Force',
            figsize=(3.2, 2.2)
        )
        
        # Shoulder Torque plot
        torque_values = 50 + 180*np.exp(-((times - 0.7)**2) / 0.02)
        torque_plot = create_time_series_plot(
            times, torque_values, std_values=np.ones_like(times)*15,
            ref_range=None,
            ylabel='Shoulder Torque (Nm)',
            title='Shoulder Torque',
            figsize=(3.2, 2.2)
        )
        
        # Elbow Torque plot
        elbow_torque_values = 20 + 110*np.exp(-((times - 0.75)**2) / 0.015)
        elbow_torque_plot = create_time_series_plot(
            times, elbow_torque_values, std_values=np.ones_like(times)*10,
            ref_range=None,
            ylabel='Elbow Varus Torque (Nm)',
            title='Elbow Torque',
            figsize=(3.2, 2.2)
        )
        
        force_img = Image(force_plot, width=3.2*inch, height=2.2*inch)
        torque_img = Image(torque_plot, width=3.2*inch, height=2.2*inch)
        elbow_torque_img = Image(elbow_torque_plot, width=3.2*inch, height=2.2*inch)
        
        # Data tables
        force_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['Max (N)', '1238N ± 66N', '1361N - 1929N'],
            ['Max (%BW)', '129% ± 7%', '176% - 224%'],
        ]
        
        shoulder_torque_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['Horiz Abd Max (Nm)', '235Nm ± 83Nm', ''],
            ['Horiz Abd Max (%BWH)', '9% ± 1%', ''],
            ['Int Rot Max (Nm)', '236Nm ± 15Nm', '151Nm - 277Nm'],
            ['Int Rot Max (%BWH)', '5% ± 1%', '17% - 29%'],
        ]
        
        elbow_torque_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['Max (Nm)', '130Nm ± 11Nm', '115Nm - 171Nm'],
            ['Max (%BWH)', '6% ± 0%', '13% - 17%'],
        ]
        
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
        ])
        
        force_table = Table(force_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch])
        force_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['red']),
        ]))
        
        shoulder_t_table = Table(shoulder_torque_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch])
        shoulder_t_table.setStyle(table_style)
        
        elbow_t_table = Table(elbow_torque_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch])
        elbow_t_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['red']),
        ]))
        
        # Layout
        layout = Table([
            [force_img, torque_img, elbow_torque_img],
            [force_table, shoulder_t_table, elbow_t_table],
        ], colWidths=[3.3*inch, 3.7*inch, 3.3*inch])
        
        layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(layout)
        
        return elements
    
    def generate_trunk_page(self):
        """Generate the trunk metrics page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Trunk</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        times = np.linspace(0, 1, 100)
        
        # Trunk Forward Tilt
        tft_values = 5 + 45*times
        tft_plot = create_time_series_plot(
            times, tft_values, std_values=np.ones_like(times)*3,
            ref_range=(9, 40),
            ylabel='Trunk Forward Tilt Angle (°)',
            title='Trunk Forward Tilt',
            figsize=(3.2, 2.2)
        )
        
        # Trunk Lateral Tilt
        tlt_values = -5 + 20*np.sin(1.5*np.pi*times)
        tlt_plot = create_time_series_plot(
            times, tlt_values, std_values=np.ones_like(times)*4,
            ref_range=(-10, 29),
            ylabel='Trunk Lateral Tilt Angle (°)',
            title='Trunk Lateral Tilt',
            figsize=(3.2, 2.2)
        )
        
        # Trunk Rotation
        tr_values = 130 - 150*times
        tr_plot = create_time_series_plot(
            times, tr_values, std_values=np.ones_like(times)*5,
            ref_range=(-24, 118),
            ylabel='Trunk Rotation Angle (°)',
            title='Trunk Rotation',
            figsize=(3.2, 2.2)
        )
        
        tft_img = Image(tft_plot, width=3.2*inch, height=2.2*inch)
        tlt_img = Image(tlt_plot, width=3.2*inch, height=2.2*inch)
        tr_img = Image(tr_plot, width=3.2*inch, height=2.2*inch)
        
        # Data tables
        tft_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '13° ± 1°', '9° - 21°'],
            ['MER', '33° ± 1°', '10° - 22°'],
            ['REL', '48° ± 1°', '26° - 40°'],
        ]
        
        tlt_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '4° ± 0.8°', '-10° - 0°'],
            ['MER', '15° ± 0.5°', '13° - 29°'],
            ['REL', '8° ± 0.5°', '8° - 28°'],
        ]
        
        tr_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '109° ± 2°', '95° - 118°'],
            ['MER', '-6° ± 0.8°', '-5° - 15°'],
            ['REL', '-20° ± 0.7°', '-4° - -24°'],
        ]
        
        tft_table = Table(tft_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        tft_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['yellow']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['yellow']),
        ]))
        
        tlt_table = Table(tlt_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        tlt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['yellow']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),
        ]))
        
        tr_table = Table(tr_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        tr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),
        ]))
        
        # Timing
        timing = Paragraph("<b>Timing:</b> On Time", 
                          ParagraphStyle('Timing', fontSize=10, textColor=COLORS['navy']))
        
        layout = Table([
            [tft_img, tlt_img, tr_img],
            [tft_table, tlt_table, tr_table],
        ], colWidths=[3.3*inch, 3.3*inch, 3.3*inch])
        
        layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(layout)
        elements.append(Spacer(1, 12))
        elements.append(timing)
        
        return elements
    
    def generate_pelvis_page(self):
        """Generate the pelvis metrics page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Pelvis</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        times = np.linspace(0, 1, 100)
        
        # Pelvic Forward Tilt
        pft_values = 5 + 55*times
        pft_plot = create_time_series_plot(
            times, pft_values, std_values=np.ones_like(times)*3,
            ref_range=(0, 56),
            ylabel='Pelvic Forward Tilt Angle (°)',
            title='Pelvic Forward Tilt',
            figsize=(3.2, 2.2)
        )
        
        # Pelvic Lateral Tilt
        plt_values = 5 - 10*times + 5*np.sin(2*np.pi*times)
        plt_plot = create_time_series_plot(
            times, plt_values, std_values=np.ones_like(times)*3,
            ref_range=(-5, 13),
            ylabel='Pelvic Lateral Tilt Angle (°)',
            title='Pelvic Lateral Tilt',
            figsize=(3.2, 2.2)
        )
        
        # Pelvic Rotation
        pr_values = 100 - 110*times
        pr_plot = create_time_series_plot(
            times, pr_values, std_values=np.ones_like(times)*5,
            ref_range=(-14, 76),
            ylabel='Pelvic Rotation Angle (°)',
            title='Pelvic Rotation',
            figsize=(3.2, 2.2)
        )
        
        pft_img = Image(pft_plot, width=3.2*inch, height=2.2*inch)
        plt_img = Image(plt_plot, width=3.2*inch, height=2.2*inch)
        pr_img = Image(pr_plot, width=3.2*inch, height=2.2*inch)
        
        # Data tables
        pft_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '7° ± 0.6°', '0° - 14°'],
            ['MER', '45° ± 1°', '32° - 46°'],
            ['REL', '55° ± 2°', '40° - 56°'],
        ]
        
        plt_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '6° ± 2°', '-3° - 5°'],
            ['MER', '2° ± 1°', '-3° - 13°'],
            ['REL', '-5° ± 2°', '-5° - 11°'],
        ]
        
        pr_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '57° ± 3°', '30° - 76°'],
            ['MER', '-9° ± 2°', '-10° - 6°'],
            ['REL', '-11° ± 2°', '-14° - 2°'],
        ]
        
        pft_table = Table(pft_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        pft_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),
        ]))
        
        plt_table = Table(plt_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        plt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['yellow']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),
        ]))
        
        pr_table = Table(pr_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch])
        pr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
            ('BACKGROUND', (1, 2), (1, 2), COLORS['green']),
            ('BACKGROUND', (1, 3), (1, 3), COLORS['green']),
        ]))
        
        timing = Paragraph("<b>Timing:</b> On Time", 
                          ParagraphStyle('Timing', fontSize=10, textColor=COLORS['navy']))
        
        layout = Table([
            [pft_img, plt_img, pr_img],
            [pft_table, plt_table, pr_table],
        ], colWidths=[3.3*inch, 3.3*inch, 3.3*inch])
        
        layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(layout)
        elements.append(Spacer(1, 12))
        elements.append(timing)
        
        return elements
    
    def generate_hip_shoulder_separation_page(self):
        """Generate the hip-shoulder separation page."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Hip-Shoulder Separation</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        times = np.linspace(0, 1, 100)
        
        # Hip-Shoulder Separation plot
        hss_values = 10 + 50*np.exp(-((times - 0.5)**2) / 0.03) - 30*times
        hss_plot = create_time_series_plot(
            times, hss_values, std_values=np.ones_like(times)*5,
            ref_range=(32, 52),
            ylabel='Hip-Shoulder Separation Angle (°)',
            title='Hip-Shoulder Separation',
            figsize=(6, 4)
        )
        
        hss_img = Image(hss_plot, width=6*inch, height=4*inch)
        
        # Description
        desc = Paragraph(
            """<b>Hip-Shoulder Separation</b> is the difference in angle created between your pelvis 
            rotation and trunk rotation. When the pelvis leads the trunk this angle is positive, and 
            when you close the gap using all the stretch you created then the angle becomes negative 
            around release as the trunk passes the pelvis. Hip-Shoulder Separation is created through 
            proper trunk and pelvis rotation timing.""",
            ParagraphStyle('Description', fontSize=10, textColor=COLORS['dark_gray'], 
                          leading=14, alignment=TA_LEFT)
        )
        
        # Data table
        hss_data = [
            ['', 'Mean ± Std Dev', 'Reference'],
            ['FP', '51° ± 2°', '32° - 52°'],
            ['MER', '11° ± 2°', ''],
            ['REL', '6° ± 3°', ''],
        ]
        
        hss_table = Table(hss_data, colWidths=[0.6*inch, 1.4*inch, 1.2*inch])
        hss_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('BACKGROUND', (1, 1), (1, 1), COLORS['green']),
        ]))
        
        # Layout
        layout = Table([
            [hss_img, desc],
            ['', hss_table],
        ], colWidths=[6.5*inch, 3.5*inch])
        
        layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(layout)
        
        return elements
    
    def generate_report_info_page(self):
        """Generate the report information page explaining graphs and abbreviations."""
        elements = []
        
        # Section header
        header = Paragraph(
            "<font color='#EAAA00' size='24'><b>Report Information</b></font>",
            ParagraphStyle('SectionHeader', alignment=TA_CENTER)
        )
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Important Events section
        events_title = Paragraph(
            "<font color='#002855' size='14'><b>Important Events</b></font>",
            ParagraphStyle('SubHeader', fontSize=14, textColor=COLORS['navy'])
        )
        
        events_text = """
        <b>MKH = Maximum Knee Height</b> and is the moment where your lead knee is at the highest point 
        in the beginning of the delivery. It is represented by the first small black tick mark on the graph.<br/><br/>
        
        <b>FP = Foot Plant</b> and is the moment where your whole lead foot contacts the ground. 
        It is represented by the vertical blue line on the graph.<br/><br/>
        
        <b>MER = Maximum External Rotation</b> and is the moment where your pitching shoulder reaches 
        its maximum external rotation angle. It is represented by the second small black tick mark on the graph.<br/><br/>
        
        <b>REL = Release</b> and is the moment where you release the ball, determined by your maximum 
        wrist velocity. It is represented by the vertical red line on the graph.
        """
        
        events_para = Paragraph(events_text, 
                               ParagraphStyle('Events', fontSize=9, textColor=COLORS['dark_gray'], leading=12))
        
        # Graph Information section
        graphs_title = Paragraph(
            "<font color='#002855' size='14'><b>Graphs</b></font>",
            ParagraphStyle('SubHeader', fontSize=14, textColor=COLORS['navy'])
        )
        
        graphs_text = """
        • All data is only on fastball mechanics<br/>
        • Graphs start 0.1 seconds before MKH and end 0.1 seconds after release<br/>
        • <b>Thin blue line</b> is the mean of all your fastball data for each metric over the duration of a pitch<br/>
        • <b>Darker blue-gray shaded area</b> surrounding the mean line is the standard deviation range 
        of all your fastball data for each metric over the duration of the pitch. The smaller the region 
        the more consistent your mechanics<br/>
        • <b>Lighter blue shaded region</b> is the reference data<br/>
        • Reference data is MLB averages from Kinatrax
        """
        
        graphs_para = Paragraph(graphs_text, 
                               ParagraphStyle('Graphs', fontSize=9, textColor=COLORS['dark_gray'], leading=12))
        
        # Abbreviations section
        abbrev_title = Paragraph(
            "<font color='#002855' size='14'><b>Other Abbreviations</b></font>",
            ParagraphStyle('SubHeader', fontSize=14, textColor=COLORS['navy'])
        )
        
        abbrev_text = """
        <b>BW = Body Weight</b> and is your weight in Newtons.<br/><br/>
        <b>BWH = Body Weight Height</b> and is your weight in Newtons multiplied by your height in meters.<br/><br/>
        <b>Std Dev = Standard Deviation</b> and is the standard deviation of your data from multiple 
        pitches away from your mean data.
        """
        
        abbrev_para = Paragraph(abbrev_text, 
                               ParagraphStyle('Abbrev', fontSize=9, textColor=COLORS['dark_gray'], leading=12))
        
        # Tables section
        tables_title = Paragraph(
            "<font color='#002855' size='14'><b>Tables</b></font>",
            ParagraphStyle('SubHeader', fontSize=14, textColor=COLORS['navy'])
        )
        
        # Color legend table
        legend_data = [
            ['Within the reference range', 'Between the reference range\nand one Std Dev outside of it', 
             'Greater than one Std Dev\noutside of the reference range'],
        ]
        
        legend_table = Table(legend_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), COLORS['green']),
            ('BACKGROUND', (1, 0), (1, 0), COLORS['yellow']),
            ('BACKGROUND', (2, 0), (2, 0), COLORS['red']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, COLORS['navy']),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        tables_note = Paragraph(
            "Being in or out of range is just a guide, it is not necessarily good or bad.",
            ParagraphStyle('Note', fontSize=9, textColor=COLORS['dark_gray'], alignment=TA_CENTER)
        )
        
        # Skeletons section
        skeletons_title = Paragraph(
            "<font color='#002855' size='14'><b>Skeletons</b></font>",
            ParagraphStyle('SubHeader', fontSize=14, textColor=COLORS['navy'])
        )
        
        skeletons_text = """
        Skeleton images are from the mechanics of your highest velocity fastball, or from your 
        first fastball if the data wasn't paired with Trackman.
        """
        
        skeletons_para = Paragraph(skeletons_text, 
                                  ParagraphStyle('Skeletons', fontSize=9, textColor=COLORS['dark_gray'], leading=12))
        
        # Create two-column layout
        left_content = [
            [events_title],
            [events_para],
            [Spacer(1, 12)],
            [graphs_title],
            [graphs_para],
        ]
        
        right_content = [
            [abbrev_title],
            [abbrev_para],
            [Spacer(1, 12)],
            [tables_title],
            [legend_table],
            [tables_note],
            [Spacer(1, 12)],
            [skeletons_title],
            [skeletons_para],
        ]
        
        left_table = Table(left_content, colWidths=[4.5*inch])
        left_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        right_table = Table(right_content, colWidths=[4.5*inch])
        right_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        layout = Table([[left_table, right_table]], colWidths=[5*inch, 5*inch])
        layout.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(layout)
        
        return elements
    
    def generate_report(self):
        """Generate the complete PDF report."""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=landscape(LETTER),
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        
        # Page 1: Summary
        elements.extend(self.generate_summary_page())
        elements.append(PageBreak())
        
        # Page 2: Shoulder
        elements.extend(self.generate_shoulder_page())
        elements.append(PageBreak())
        
        # Page 3: Elbow / Arm Slot
        elements.extend(self.generate_elbow_arm_slot_page())
        elements.append(PageBreak())
        
        # Page 4: Throwing Arm Stress
        elements.extend(self.generate_stress_page())
        elements.append(PageBreak())
        
        # Page 5: Trunk
        elements.extend(self.generate_trunk_page())
        elements.append(PageBreak())
        
        # Page 6: Pelvis
        elements.extend(self.generate_pelvis_page())
        elements.append(PageBreak())
        
        # Page 7: Hip-Shoulder Separation
        elements.extend(self.generate_hip_shoulder_separation_page())
        elements.append(PageBreak())
        
        # Page 8: Kinematic Sequence
        elements.extend(self.generate_kinematic_sequence_page())
        elements.append(PageBreak())
        
        # Page 9: Report Information
        elements.extend(self.generate_report_info_page())
        
        # Build PDF
        doc.build(elements)
        
        print(f"Report generated: {self.output_path}")
        return self.output_path


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_process_data():
    """Load and process the pitching biomechanics data."""
    print("Using sample data for Luke Coats mocap report...")
    
    # Sample data based on the PDF provided (Luke Coats, August 29, 2025)
    # In production, this would load from actual mocap data files
    sample_data = {
        'pitch_speed_mph': [90, 91, 92, 91, 90],
        'p_throws': ['R', 'R', 'R', 'R', 'R'],
        'arm_slot': [62, 62, 62, 62, 62],
        'shoulder_horizontal_abduction_fp': [-44, -45, -43, -44, -44],
        'shoulder_horizontal_abduction_mer': [-4, -6, -2, -4, -4],
        'shoulder_horizontal_abduction_rel': [1, 0, 2, 1, 1],
        'shoulder_abduction_fp': [91, 89, 93, 91, 91],
        'shoulder_abduction_mer': [102, 101.4, 102.6, 102, 102],
        'shoulder_abduction_rel': [99, 98, 100, 99, 99],
        'shoulder_external_rotation_fp': [48, 44, 52, 48, 48],
        'shoulder_external_rotation_mer': [193, 192, 194, 193, 193],
        'shoulder_external_rotation_rel': [123, 120, 126, 123, 123],
        'elbow_flexion_fp': [113, 112, 114, 113, 113],
        'elbow_flexion_mer': [86, 85, 87, 86, 86],
        'elbow_flexion_rel': [30, 29, 31, 30, 30],
        'trunk_forward_tilt_fp': [13, 12, 14, 13, 13],
        'trunk_forward_tilt_mer': [33, 32, 34, 33, 33],
        'trunk_forward_tilt_rel': [48, 47, 49, 48, 48],
        'trunk_rotation_fp': [109, 107, 111, 109, 109],
        'trunk_rotation_mer': [-6, -6.8, -5.2, -6, -6],
        'trunk_rotation_rel': [-20, -20.7, -19.3, -20, -20],
        'hip_shoulder_separation_fp': [51, 49, 53, 51, 51],
        'hip_shoulder_separation_mer': [11, 9, 13, 11, 11],
        'hip_shoulder_separation_rel': [6, 3, 9, 6, 6],
        'pelvic_forward_tilt_fp': [7, 6.4, 7.6, 7, 7],
        'pelvic_forward_tilt_mer': [45, 44, 46, 45, 45],
        'pelvic_forward_tilt_rel': [55, 53, 57, 55, 55],
        'pelvic_rotation_fp': [57, 54, 60, 57, 57],
        'pelvic_rotation_mer': [-9, -11, -7, -9, -9],
        'pelvic_rotation_rel': [-11, -13, -9, -11, -11],
        'knee_flexion_fp': [61, 59, 63, 61, 61],
        'knee_flexion_mer': [58, 54, 62, 58, 58],
        'knee_flexion_rel': [48, 45, 51, 48, 48],
        'pelvis_angular_velocity': [594, 574, 614, 594, 594],
        'trunk_angular_velocity': [1060, 1027, 1093, 1060, 1060],
        'elbow_angular_velocity': [2209, 2151, 2267, 2209, 2209],
        'shoulder_angular_velocity': [4277, 4115, 4439, 4277, 4277],
        'knee_angular_velocity': [-338, -392, -284, -338, -338],
        'shoulder_force_bw': [129, 122, 136, 129, 129],
        'shoulder_torque_bwh': [7, 6, 8, 7, 7],
        'elbow_torque_bwh': [6, 6, 6, 6, 6],
    }
    
    pitching_poi_filtered = pd.DataFrame(sample_data)
    
    print(f"Loaded {len(pitching_poi_filtered)} sample pitches")
    
    # Calculate statistics
    stats = []
    for col in pitching_poi_filtered.columns:
        if pd.api.types.is_numeric_dtype(pitching_poi_filtered[col]):
            stats.append({
                'column': col,
                'mean': pitching_poi_filtered[col].mean(),
                'std': pitching_poi_filtered[col].std()
            })
    
    stats_df = pd.DataFrame(stats)
    
    return pitching_poi_filtered, stats_df


def main():
    """Main function to generate the mocap report."""
    # Load data
    pitching_data, stats = load_and_process_data()
    
    # Use current directory for output (works on any machine)
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "mocap_report.pdf")
    
    # Create report generator
    generator = MocapReportGenerator(
        player_name="Luke Coats",
        date="August 29, 2025",
        velocity_range="90mph - 92mph",
        output_path=output_path
    )
    
    # Set sample metrics (in a real scenario, these would come from your mocap data)
    metrics = {
        'angular_velocities': {
            'knee': -338,
            'pelvis': 594,
            'trunk': 1060,
            'elbow': 2209,
            'shoulder': 4277,
        },
        'stress': {
            'shoulder_force': 129,
            'shoulder_torque': 7,
            'elbow_torque': 6,
        },
        'arm_slot': (62, 2),
    }
    
    generator.set_metrics(metrics)
    
    # Generate the report
    output_path = generator.generate_report()
    
    return output_path


if __name__ == "__main__":
    main()