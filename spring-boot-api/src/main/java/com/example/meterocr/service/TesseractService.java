
package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;

@Service
public class TesseractService {
    private final String tessdataPrefix;

    public TesseractService(@Value("${app.tessdata-prefix:}") String tessdataPrefix) {
        this.tessdataPrefix = tessdataPrefix;
    }

    private Tesseract create(MeterProfile.TessConfig cfg) {
        Tesseract t = new Tesseract();
        if (tessdataPrefix != null && !tessdataPrefix.isBlank()) {
            t.setDatapath(tessdataPrefix);
        }
        t.setLanguage("eng");
        t.setTessVariable("tessedit_char_whitelist", cfg.whitelist);
        t.setTessVariable("user_defined_dpi", String.valueOf(cfg.dpi));
        t.setPageSegMode(cfg.psm);
        t.setOcrEngineMode(1); // LSTM only
        if (cfg.disable_dawg) {
            t.setTessVariable("load_system_dawg", "0");
            t.setTessVariable("load_freq_dawg", "0");
        }
        return t;
    }

    public String ocrDigits(BufferedImage img, MeterProfile profile) throws TesseractException {
        Tesseract t = create(profile.tesseract);
        return t.doOCR(img);
    }
}
