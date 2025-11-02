
package com.example.meterocr.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RegexUtils {
    public static final Pattern READING = Pattern.compile(
            "(?i)\\b(\\w{1,9}(?:[.,]\\d{1,3})?)(?!\\d)\\s*[-_\\s]*?(điện|dien|kwh|kw|volt|v|kilowatt|nước|nuoc|m3|lit|lít|liter|khối|khoi|){3,8}\\b"
    );
    //    private static final Pattern SERIAL = Pattern.compile("\\b[A-Z0-9-]{6,}\\b");
    public static final Pattern SERIAL = Pattern.compile("(?i)\\b(?:Serial|SX|SR)?[A-Z0-9-]{6,}\\b");

    private static final Pattern METER_UNIT = Pattern.compile("(?i)\\b(điện|dien|kwh|kw|volt|v|kilowatt|nước|nuoc|m3|lit|lít|liter|khối|khoi)\\b");

    public static String normalize(String s) {
        if (s == null) return null;
        String t = s.trim();
        t = t.replace('O', '0').replace('o', '0');
        t = t.replace('I', '1').replace('l', '1');
        if (t.indexOf(',') >= 0 && t.indexOf('.') < 0) t = t.replace(',', '.');
        return t;
    }

    public static boolean looksLikeReading(String s) {
        if (s == null || s.length() < 3 || s.length() > 9) return false;
        return READING.matcher(s).find();
    }

    public static boolean looksLikeSerial(String s) {
        if (s == null) return false;
        return SERIAL.matcher(s).find();
    }

    public static String extractUnit(String s) {
        if (s == null) return "";
        Matcher matcher = METER_UNIT.matcher(s);
        if (matcher.find()) {
            return matcher.group();
        }
        return "";
    }

    public static String extractBestReading(String text) {
        if (text == null) return null;
        String norm = normalize(text);
        Matcher m = READING.matcher(norm);
        String best = null;
        while (m.find()) {
            String cand = m.group(1);
            if (best == null || cand.length() > best.length()) best = cand;
        }
        return best;
    }
}
