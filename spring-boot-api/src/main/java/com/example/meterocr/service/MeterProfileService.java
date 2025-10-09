
package com.example.meterocr.service;

import com.example.meterocr.model.MeterProfile;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

@Service
public class MeterProfileService {
    private final Map<String, MeterProfile> profiles = new HashMap<>();

    public MeterProfileService() {
        try {
            ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
            InputStream is = new ClassPathResource("meter-profiles.yml").getInputStream();
            JsonNode root = mapper.readTree(is);
            JsonNode pnode = root.get("profiles");
            if (pnode != null) {
                Iterator<String> it = pnode.fieldNames();
                while (it.hasNext()) {
                    String name = it.next();
                    MeterProfile mp = mapper.treeToValue(pnode.get(name), MeterProfile.class);
                    profiles.put(name, mp);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load meter-profiles.yml", e);
        }
        if (!profiles.containsKey("lcd_digital")) {
            profiles.put("lcd_digital", new MeterProfile());
        }
    }

    public MeterProfile get(String name) {
        if (name == null || name.isBlank()) return profiles.get("lcd_digital");
        return profiles.getOrDefault(name, profiles.get("lcd_digital"));
    }

    public Map<String, MeterProfile> all() { return profiles; }
}
