//! Minimal EV3 framebuffer display for gyro diagnostics.

use ev3dev_lang_rust::{Ev3Result, Screen};
use image::Rgb;

const BLACK: Rgb<u8> = Rgb([0, 0, 0]);
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const SCALE: u32 = 2;

pub struct Display {
    screen: Screen,
}

impl Display {
    pub fn new() -> Ev3Result<Self> {
        let mut screen = Screen::new()?;
        screen.clear();
        screen.update();
        Ok(Self { screen })
    }

    pub fn show_gyro(&mut self, angle_rad: f32, rate_rad_s: f32) {
        self.screen.clear();
        draw_text(&mut self.screen, 4, 8, "ANGLE", SCALE);
        draw_text(
            &mut self.screen,
            4,
            26,
            &format_value(angle_rad),
            SCALE,
        );
        draw_text(&mut self.screen, 4, 62, "RATE", SCALE);
        draw_text(
            &mut self.screen,
            4,
            80,
            &format_value(rate_rad_s),
            SCALE,
        );
        self.screen.update();
    }
}

fn format_value(value: f32) -> String {
    format!("{value:+.2}")
}

fn draw_text(screen: &mut Screen, x: u32, y: u32, text: &str, scale: u32) {
    let mut cursor = x;
    for character in text.chars() {
        draw_char(screen, cursor, y, character, scale);
        cursor += 6 * scale;
    }
}

fn draw_char(screen: &mut Screen, x: u32, y: u32, character: char, scale: u32) {
    let glyph = match character {
        'A' => ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        'E' => ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        'G' => ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
        'L' => ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        'N' => ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        'R' => ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        'T' => ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        '+' => ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
        '-' => ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        '.' => ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
        '0' => ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        '1' => ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        '2' => ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        '3' => ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        '4' => ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        '5' => ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
        '6' => ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        '7' => ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        '8' => ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        '9' => ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        _ => ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    };

    for (row, pattern) in glyph.iter().enumerate() {
        for (column, pixel) in pattern.bytes().enumerate() {
            if pixel != b'1' {
                continue;
            }
            for dy in 0..scale {
                for dx in 0..scale {
                    let px = x + column as u32 * scale + dx;
                    let py = y + row as u32 * scale + dy;
                    if px < screen.xres() && py < screen.yres() {
                        screen.image.put_pixel(px, py, BLACK);
                    }
                }
            }
        }
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        self.screen.clear();
        self.screen.image.put_pixel(0, 0, WHITE);
        self.screen.update();
    }
}