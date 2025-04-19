import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.util.Base64;
import java.util.concurrent.*;
import org.json.*;
import com.github.sarxos.webcam.*;
import javax.imageio.ImageIO;

public class GestureControlClient extends JFrame {
    // Configuration
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 8765;
    private static final int FRAME_WIDTH = 1024;
    private static final int FRAME_HEIGHT = 600;
    private static final int CAM_WIDTH = 640;
    private static final int CAM_HEIGHT = 480;
    private static final int UI_WIDTH = FRAME_WIDTH - CAM_WIDTH;
    private static final int PANEL_PADDING = 20;
    private static final int GESTURE_DISPLAY_TIME = 2000; // Show gesture label for 2 seconds

    // Modern Dark Theme Colors
    private static final Color BG_COLOR = new Color(18, 18, 26);
    private static final Color PANEL_COLOR = new Color(24, 24, 37);
    private static final Color PRIMARY_COLOR = new Color(80, 250, 123);
    private static final Color ACCENT_COLOR = new Color(255, 184, 108);
    private static final Color SECONDARY_COLOR = new Color(139, 233, 253);
    private static final Color TEXT_COLOR = new Color(248, 248, 242);
    private static final Color DARK_TEXT = new Color(68, 71, 90);
    private static final Color INACTIVE_COLOR = new Color(98, 114, 164);
    private static final Color BUTTON_HOVER = new Color(40, 42, 54);

    // UI Components
    private JPanel cameraPanel;
    private JPanel controlPanel;
    private CustomProgressBar fanMeter;
    private CustomProgressBar volumeMeter;
    private RoundedButton toggleFanButton;
    private RoundedButton increaseButton;
    private RoundedButton decreaseButton;
    private JLabel fanStatusLabel;
    private JLabel volumeStatusLabel;
    private JLabel connectionStatusLabel;
    private JLabel fpsLabel;
    private JLabel gestureLabel;

    // State
    private boolean fanState = false;
    private int fanSpeed = 100;
    private int lastFanSpeed = 100;
    private int volumeLevel = 50;
    private int currentFps = 0;
    private String lastGesture = "";
    private long lastGestureTime = 0;

    // WebSocket
    private WebSocket webSocket;
    private boolean connected = false;
    private ScheduledExecutorService scheduler;

    // Webcam
    private Webcam webcam;
    private WebcamPanel webcamPanel;
    private boolean isStreaming = false;
    private BufferedImage currentFrame;
    private final Object frameLock = new Object();

    public GestureControlClient() {
        super("Gesture Control System");
        setSize(FRAME_WIDTH, FRAME_HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        setBackground(BG_COLOR);
        setIconImage(createAppIcon());

        // Initialize components
        initializeUI();

        // Connect to WebSocket
        connectWebSocket();

        // Initialize webcam
        initializeWebcam();

        // Start frame sender
        startFrameSender();

        setLocationRelativeTo(null); // Center on screen
        setVisible(true);
    }

    private Image createAppIcon() {
        BufferedImage icon = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = icon.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setColor(PANEL_COLOR);
        g2d.fillRect(0, 0, 32, 32);
        g2d.setColor(PRIMARY_COLOR);
        g2d.fillOval(4, 4, 24, 24);
        g2d.setColor(PANEL_COLOR);
        g2d.fillOval(8, 8, 16, 16);
        g2d.dispose();
        return icon;
    }

    private void initializeUI() {
        // Set background for the frame
        JPanel contentPane = new JPanel(new BorderLayout());
        contentPane.setBackground(BG_COLOR);
        setContentPane(contentPane);

        // Camera panel (left side)
        cameraPanel = new JPanel();
        cameraPanel.setPreferredSize(new Dimension(CAM_WIDTH, FRAME_HEIGHT));
        cameraPanel.setBackground(Color.BLACK);
        cameraPanel.setLayout(new BorderLayout());
        cameraPanel.setBorder(BorderFactory.createMatteBorder(0, 0, 0, 2, PANEL_COLOR.darker()));

        // FPS and Gesture labels
        JPanel overlayPanel = new JPanel(new BorderLayout());
        overlayPanel.setOpaque(false);

        fpsLabel = new JLabel("FPS: 0");
        fpsLabel.setForeground(SECONDARY_COLOR);
        fpsLabel.setFont(new Font("Segoe UI", Font.BOLD, 14));
        fpsLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 0, 0));
        overlayPanel.add(fpsLabel, BorderLayout.NORTH);

        gestureLabel = new JLabel("");
        gestureLabel.setForeground(ACCENT_COLOR);
        gestureLabel.setFont(new Font("Segoe UI", Font.BOLD, 18));
        gestureLabel.setHorizontalAlignment(JLabel.CENTER);
        gestureLabel.setBorder(BorderFactory.createCompoundBorder(
                new RoundedBorder(PANEL_COLOR, 10),
                BorderFactory.createEmptyBorder(5, 10, 5, 10)
        ));
        gestureLabel.setOpaque(false);
        gestureLabel.setVisible(false);  // Hide until a gesture is detected

        JPanel gestureLabelPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        gestureLabelPanel.setOpaque(false);
        gestureLabelPanel.add(gestureLabel);
        overlayPanel.add(gestureLabelPanel, BorderLayout.SOUTH);

        cameraPanel.add(overlayPanel, BorderLayout.CENTER);

        // Control panel (right side)
        controlPanel = new RoundedPanel(20, PANEL_COLOR);
        controlPanel.setPreferredSize(new Dimension(UI_WIDTH, FRAME_HEIGHT));
        controlPanel.setLayout(null); // Absolute positioning

        // Title
        JLabel titleLabel = new JLabel("Gesture Control");
        titleLabel.setForeground(TEXT_COLOR);
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 24));
        titleLabel.setBounds(PANEL_PADDING, 10, UI_WIDTH - 2 * PANEL_PADDING, 30);
        controlPanel.add(titleLabel);

        // Fan meter section
        RoundedPanel fanPanel = new RoundedPanel(15, PANEL_COLOR.darker());
        fanPanel.setBounds(PANEL_PADDING, 50, UI_WIDTH - 2 * PANEL_PADDING, 220);
        fanPanel.setLayout(null);

        JLabel fanLabel = new JLabel("Fan Control (Right Hand)");
        fanLabel.setForeground(TEXT_COLOR);
        fanLabel.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        fanLabel.setBounds(15, 10, UI_WIDTH - 2 * PANEL_PADDING - 30, 20);
        fanPanel.add(fanLabel);

        fanMeter = new CustomProgressBar(0, 100, PRIMARY_COLOR, INACTIVE_COLOR, PANEL_COLOR);
        fanMeter.setValue(fanSpeed);
        fanMeter.setOrientation(SwingConstants.VERTICAL);
        fanMeter.setBounds((UI_WIDTH - 2 * PANEL_PADDING) / 2 - 40, 40, 80, 120);
        fanPanel.add(fanMeter);

        fanStatusLabel = new JLabel("Fan: STOPPED");
        fanStatusLabel.setForeground(INACTIVE_COLOR);
        fanStatusLabel.setFont(new Font("Segoe UI", Font.BOLD, 16));
        fanStatusLabel.setHorizontalAlignment(JLabel.CENTER);
        fanStatusLabel.setBounds(0, 170, UI_WIDTH - 2 * PANEL_PADDING, 20);
        fanPanel.add(fanStatusLabel);

        controlPanel.add(fanPanel);

        // Volume meter section
        RoundedPanel volumePanel = new RoundedPanel(15, PANEL_COLOR.darker());
        volumePanel.setBounds(PANEL_PADDING, 280, UI_WIDTH - 2 * PANEL_PADDING, 220);
        volumePanel.setLayout(null);

        JLabel volumeLabel = new JLabel("Volume Control (Left Hand)");
        volumeLabel.setForeground(TEXT_COLOR);
        volumeLabel.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        volumeLabel.setBounds(15, 10, UI_WIDTH - 2 * PANEL_PADDING - 30, 20);
        volumePanel.add(volumeLabel);

        volumeMeter = new CustomProgressBar(0, 100, SECONDARY_COLOR, INACTIVE_COLOR, PANEL_COLOR);
        volumeMeter.setValue(volumeLevel);
        volumeMeter.setOrientation(SwingConstants.VERTICAL);
        volumeMeter.setBounds((UI_WIDTH - 2 * PANEL_PADDING) / 2 - 40, 40, 80, 120);
        volumePanel.add(volumeMeter);

        volumeStatusLabel = new JLabel("System Volume: 50%");
        volumeStatusLabel.setForeground(SECONDARY_COLOR);
        volumeStatusLabel.setFont(new Font("Segoe UI", Font.BOLD, 16));
        volumeStatusLabel.setHorizontalAlignment(JLabel.CENTER);
        volumeStatusLabel.setBounds(0, 170, UI_WIDTH - 2 * PANEL_PADDING, 20);
        volumePanel.add(volumeStatusLabel);

        controlPanel.add(volumePanel);

        // Control buttons
        int buttonWidth = 50;
        int buttonHeight = 50;
        int buttonSpacing = 10;
        int totalButtonsWidth = 2 * buttonWidth + buttonSpacing;
        int buttonsStartX = PANEL_PADDING + (UI_WIDTH - 2 * PANEL_PADDING - totalButtonsWidth) / 2;

        increaseButton = new RoundedButton("+", buttonWidth, buttonHeight);
        increaseButton.setBounds(buttonsStartX, 510, buttonWidth, buttonHeight);
        increaseButton.setFont(new Font("Segoe UI", Font.BOLD, 20));
        increaseButton.setForeground(TEXT_COLOR);
        increaseButton.setBackground(INACTIVE_COLOR);
        increaseButton.setEnabled(false);
        increaseButton.addActionListener(e -> adjustFanSpeed(25));
        controlPanel.add(increaseButton);

        decreaseButton = new RoundedButton("-", buttonWidth, buttonHeight);
        decreaseButton.setBounds(buttonsStartX + buttonWidth + buttonSpacing, 510, buttonWidth, buttonHeight);
        decreaseButton.setFont(new Font("Segoe UI", Font.BOLD, 20));
        decreaseButton.setForeground(TEXT_COLOR);
        decreaseButton.setBackground(INACTIVE_COLOR);
        decreaseButton.setEnabled(false);
        decreaseButton.addActionListener(e -> adjustFanSpeed(-25));
        controlPanel.add(decreaseButton);

        toggleFanButton = new RoundedButton("TOGGLE FAN", UI_WIDTH - 2 * PANEL_PADDING, 50);
        toggleFanButton.setBounds(PANEL_PADDING, 570, UI_WIDTH - 2 * PANEL_PADDING, 50);
        toggleFanButton.setFont(new Font("Segoe UI", Font.BOLD, 14));
        toggleFanButton.setForeground(TEXT_COLOR);
        toggleFanButton.setBackground(INACTIVE_COLOR);
        toggleFanButton.addActionListener(e -> toggleFan());
        controlPanel.add(toggleFanButton);

        // Connection status
        connectionStatusLabel = new JLabel("Connection: Disconnected");
        connectionStatusLabel.setForeground(ACCENT_COLOR);
        connectionStatusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        connectionStatusLabel.setBounds(PANEL_PADDING, FRAME_HEIGHT - 40, UI_WIDTH - 2 * PANEL_PADDING, 20);
        controlPanel.add(connectionStatusLabel);

        // Add panels to frame
        JPanel wrapperPanel = new JPanel(new BorderLayout());
        wrapperPanel.setBackground(BG_COLOR);
        wrapperPanel.add(cameraPanel, BorderLayout.WEST);

        JPanel controlWrapperPanel = new JPanel(new BorderLayout());
        controlWrapperPanel.setBackground(BG_COLOR);
        controlWrapperPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        controlWrapperPanel.add(controlPanel, BorderLayout.CENTER);

        wrapperPanel.add(controlWrapperPanel, BorderLayout.CENTER);

        add(wrapperPanel, BorderLayout.CENTER);
    }

    private void initializeWebcam() {
        try {
            webcam = Webcam.getDefault();
            if (webcam != null) {
                webcam.setViewSize(new Dimension(CAM_WIDTH, CAM_HEIGHT));
                webcam.open();

                webcamPanel = new WebcamPanel(webcam);
                webcamPanel.setFPSDisplayed(false);
                webcamPanel.setDisplayDebugInfo(false);
                webcamPanel.setImageSizeDisplayed(false);
                webcamPanel.setMirrored(true);

                cameraPanel.add(webcamPanel, BorderLayout.CENTER);

                // Start frame capturing thread
                new Thread(() -> {
                    while (true) {
                        if (webcam.isOpen() && connected) {
                            BufferedImage frame = webcam.getImage();
                            if (frame != null) {
                                synchronized (frameLock) {
                                    currentFrame = frame;
                                }
                            }
                        }
                        try {
                            Thread.sleep(33); // ~30 FPS
                        } catch (InterruptedException e) {
                            break;
                        }
                    }
                }).start();

                isStreaming = true;
            } else {
                JLabel errorLabel = new JLabel("No webcam detected!");
                errorLabel.setForeground(Color.RED);
                errorLabel.setHorizontalAlignment(JLabel.CENTER);
                cameraPanel.add(errorLabel, BorderLayout.CENTER);
            }
        } catch (Exception e) {
            e.printStackTrace();
            JLabel errorLabel = new JLabel("Error initializing webcam: " + e.getMessage());
            errorLabel.setForeground(Color.RED);
            errorLabel.setHorizontalAlignment(JLabel.CENTER);
            cameraPanel.add(errorLabel, BorderLayout.CENTER);
        }
    }

    private void connectWebSocket() {
        try {
            HttpClient client = HttpClient.newHttpClient();
            WebSocket.Listener listener = new WebSocketListener();

            CompletableFuture<WebSocket> futureWs = client.newWebSocketBuilder()
                    .buildAsync(URI.create("ws://" + SERVER_HOST + ":" + SERVER_PORT), listener);

            webSocket = futureWs.get(5, TimeUnit.SECONDS);

            // Start a scheduler to check connection status
            scheduler = Executors.newSingleThreadScheduledExecutor();
            scheduler.scheduleAtFixedRate(() -> {
                if (webSocket != null) {
                    try {
                        // Send a heartbeat to keep connection alive
                        webSocket.sendText("{\"type\":\"heartbeat\"}", true);
                    } catch (Exception e) {
                        connected = false;
                        updateConnectionStatus();
                        // Try to reconnect
                        reconnectWebSocket();
                    }
                }
            }, 5, 5, TimeUnit.SECONDS);

        } catch (Exception e) {
            e.printStackTrace();
            connectionStatusLabel.setText("Connection: Failed - " + e.getMessage());
            // Schedule reconnection attempt
            if (scheduler == null) {
                scheduler = Executors.newSingleThreadScheduledExecutor();
            }
            scheduler.schedule(this::reconnectWebSocket, 5, TimeUnit.SECONDS);
        }
    }

    private void reconnectWebSocket() {
        SwingUtilities.invokeLater(() -> {
            connectionStatusLabel.setText("Connection: Reconnecting...");
        });

        try {
            if (webSocket != null) {
                try {
                    webSocket.abort();
                } catch (Exception ignored) {}
            }

            HttpClient client = HttpClient.newHttpClient();
            WebSocket.Listener listener = new WebSocketListener();

            CompletableFuture<WebSocket> futureWs = client.newWebSocketBuilder()
                    .buildAsync(URI.create("ws://" + SERVER_HOST + ":" + SERVER_PORT), listener);

            webSocket = futureWs.get(5, TimeUnit.SECONDS);
        } catch (Exception e) {
            e.printStackTrace();
            SwingUtilities.invokeLater(() -> {
                connectionStatusLabel.setText("Connection: Failed - Retrying in 5s");
            });

            scheduler.schedule(this::reconnectWebSocket, 5, TimeUnit.SECONDS);
        }
    }

    private void startFrameSender() {
        ScheduledExecutorService frameScheduler = Executors.newSingleThreadScheduledExecutor();
        frameScheduler.scheduleAtFixedRate(() -> {
            if (connected && isStreaming && webcam != null && webcam.isOpen()) {
                try {
                    BufferedImage frame;
                    synchronized (frameLock) {
                        if (currentFrame == null) return;
                        frame = currentFrame;
                    }

                    // Convert frame to JPEG
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ImageIO.write(frame, "jpg", baos);
                    byte[] jpegData = baos.toByteArray();

                    // Send frame to server
                    if (webSocket != null) {
                        byte[] header = "frame:".getBytes();
                        ByteBuffer buffer = ByteBuffer.allocate(header.length + jpegData.length);
                        buffer.put(header);
                        buffer.put(jpegData);
                        buffer.flip();

                        webSocket.sendBinary(buffer, true).join();
                    }
                } catch (Exception e) {
                    System.err.println("Error sending frame: " + e.getMessage());
                }
            }
        }, 0, 100, TimeUnit.MILLISECONDS); // Send frames at 10fps
    }

    private void updateUI(JSONObject state) {
        SwingUtilities.invokeLater(() -> {
            try {
                if (state.has("fan_state")) {
                    fanState = state.getBoolean("fan_state");
                }

                if (state.has("fan_speed")) {
                    fanSpeed = state.getInt("fan_speed");
                    if (!fanState) {
                        lastFanSpeed = fanSpeed > 0 ? fanSpeed : lastFanSpeed;
                    }
                }

                if (state.has("volume")) {
                    volumeLevel = state.getInt("volume");
                }

                // Update UI components
                fanMeter.setValue(fanSpeed);
                fanMeter.setActiveColor(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);
                fanStatusLabel.setText("Fan: " + (fanState ? "RUNNING" : "STOPPED"));
                fanStatusLabel.setForeground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);

                volumeMeter.setValue(volumeLevel);
                volumeStatusLabel.setText("System Volume: " + volumeLevel + "%");

                toggleFanButton.setBackground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);

                increaseButton.setEnabled(fanState);
                increaseButton.setBackground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);
                decreaseButton.setEnabled(fanState);
                decreaseButton.setBackground(fanState ? PRIMARY_COLOR : INACTIVE_COLOR);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private void updateConnectionStatus() {
        SwingUtilities.invokeLater(() -> {
            connectionStatusLabel.setText("Connection: " + (connected ? "Connected" : "Disconnected"));
            connectionStatusLabel.setForeground(connected ? PRIMARY_COLOR : Color.RED);
        });
    }

    private void updateGestureDisplay(String gesture) {
        SwingUtilities.invokeLater(() -> {
            lastGesture = gesture;
            lastGestureTime = System.currentTimeMillis();

            // Make gesture name more readable
            String displayText = "";
            switch (gesture) {
                case "palm-open":
                    displayText = "PALM OPEN - Fan ON";
                    break;
                case "palm-closed":
                    displayText = "PALM CLOSED - Fan OFF";
                    break;
                case "pinch":
                    displayText = "PINCH - Decrease";
                    break;
                case "spread":
                    displayText = "SPREAD - Increase";
                    break;
                default:
                    displayText = gesture.toUpperCase();
            }

            gestureLabel.setText(displayText);
            gestureLabel.setVisible(true);

            // Start a timer to clear the gesture after 2 seconds
            Timer timer = new Timer(GESTURE_DISPLAY_TIME, e -> {
                if (lastGesture.equals(gesture) &&
                        System.currentTimeMillis() - lastGestureTime >= GESTURE_DISPLAY_TIME) {
                    gestureLabel.setText("");
                    gestureLabel.setVisible(false);
                }
            });
            timer.setRepeats(false);
            timer.start();
        });
    }

    private void toggleFan() {
        try {
            fanState = !fanState;

            // Update fan speed based on state
            if (fanState) {
                fanSpeed = lastFanSpeed > 0 ? lastFanSpeed : 100;
            } else {
                lastFanSpeed = fanSpeed;
                fanSpeed = 0;
            }

            // Send command to server
            JSONObject command = new JSONObject();
            command.put("command", "set_fan");
            command.put("state", fanState);
            command.put("value", fanSpeed);

            webSocket.sendText(command.toString(), true);

            // Update UI immediately for responsiveness
            updateUI(command);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void adjustFanSpeed(int delta) {
        try {
            int newSpeed = Math.max(0, Math.min(100, fanSpeed + delta));
            if (newSpeed != fanSpeed) {
                fanSpeed = newSpeed;

                // Send command to server
                JSONObject command = new JSONObject();
                command.put("command", "set_fan");
                command.put("state", fanState);
                command.put("value", fanSpeed);

                webSocket.sendText(command.toString(), true);

                // Update UI immediately
                updateUI(command);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private class WebSocketListener implements WebSocket.Listener {
        private StringBuilder textBuffer = new StringBuilder();

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            textBuffer.append(data);

            if (last) {
                try {
                    String message = textBuffer.toString();
                    JSONObject json = new JSONObject(message);

                    String type = json.optString("type", "");

                    if ("state".equals(type)) {
                        // State update
                        updateUI(json);
                    } else if ("gesture".equals(type)) {
                        // Gesture detected
                        String gestureName = json.optString("name", "");
                        if (!gestureName.isEmpty()) {
                            updateGestureDisplay(gestureName);
                        }
                    }
                } catch (JSONException e) {
                    // Not JSON or invalid format, ignore
                    System.err.println("Received non-JSON text: " + textBuffer);
                }

                textBuffer = new StringBuilder();
            }

            return WebSocket.Listener.super.onText(webSocket, data, last);
        }

        @Override
        public void onOpen(WebSocket webSocket) {
            connected = true;
            updateConnectionStatus();
            WebSocket.Listener.super.onOpen(webSocket);
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            connected = false;
            updateConnectionStatus();
            // Try to reconnect
            scheduler.schedule(GestureControlClient.this::reconnectWebSocket, 5, TimeUnit.SECONDS);
            return WebSocket.Listener.super.onClose(webSocket, statusCode, reason);
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            error.printStackTrace();
            connected = false;
            updateConnectionStatus();
            WebSocket.Listener.super.onError(webSocket, error);
        }
    }

    // Custom UI Components

    // Custom rounded panel
    private static class RoundedPanel extends JPanel {
        private int cornerRadius;
        private Color backgroundColor;

        public RoundedPanel(int radius, Color bgColor) {
            super();
            cornerRadius = radius;
            backgroundColor = bgColor;
            setOpaque(false);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Dimension arcs = new Dimension(cornerRadius, cornerRadius);
            int width = getWidth();
            int height = getHeight();
            Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Draw background
            g2d.setColor(backgroundColor);
            g2d.fillRoundRect(0, 0, width-1, height-1, arcs.width, arcs.height);
        }
    }

    // Custom rounded border
    private static class RoundedBorder implements Border {
        private int radius;
        private Color color;

        public RoundedBorder(Color color, int radius) {
            this.radius = radius;
            this.color = color;
        }

        @Override
        public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setColor(color);
            g2d.drawRoundRect(x, y, width-1, height-1, radius, radius);
            g2d.dispose();
        }

        @Override
        public Insets getBorderInsets(Component c) {
            return new Insets(radius/2, radius/2, radius/2, radius/2);
        }

        @Override
        public boolean isBorderOpaque() {
            return false;
        }
    }

    // Custom rounded button
    private static class RoundedButton extends JButton {
        private int cornerRadius = 15;
        private boolean isHovered = false;

        public RoundedButton(String text, int width, int height) {
            super(text);
            setOpaque(false);
            setContentAreaFilled(false);
            setFocusPainted(false);
            setBorderPainted(false);
            setPreferredSize(new Dimension(width, height));

            // Add hover effect
            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseEntered(MouseEvent e) {
                    if (isEnabled()) {
                        isHovered = true;
                        repaint();
                    }
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    isHovered = false;
                    repaint();
                }
            });
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Draw background
            if (isEnabled()) {
                g2d.setColor(isHovered ? getBackground().brighter() : getBackground());
            } else {
                g2d.setColor(INACTIVE_COLOR.darker());
            }

            g2d.fillRoundRect(0, 0, getWidth(), getHeight(), cornerRadius, cornerRadius);

            // Draw border
            g2d.setColor(isEnabled() ? getBackground().brighter() : INACTIVE_COLOR);
            g2d.drawRoundRect(0, 0, getWidth()-1, getHeight()-1, cornerRadius, cornerRadius);

            g2d.dispose();

            super.paintComponent(g);
        }
    }

    // Custom progress bar
    private static class CustomProgressBar extends JProgressBar {
        private Color activeColor;
        private Color inactiveColor;
        private Color backgroundColor;

        public CustomProgressBar(int min, int max, Color active, Color inactive, Color bg) {
            super(min, max);
            this.activeColor = active;
            this.inactiveColor = inactive;
            this.backgroundColor = bg;
            setOpaque(false);
            setBorderPainted(false);
            setForeground(active);
        }

        public void setActiveColor(Color color) {
            this.activeColor = color;
            setForeground(color);
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int cornerRadius = 10;
            int width = getWidth();
            int height = getHeight();

            // Background
            g2d.setColor(backgroundColor);
            g2d.fillRoundRect(0, 0, width, height, cornerRadius, cornerRadius);

            // Border
            g2d.setColor(inactiveColor.darker());
            g2d.drawRoundRect(0, 0, width-1, height-1, cornerRadius, cornerRadius);

            // Progress
            if (getOrientation() == SwingConstants.HORIZONTAL) {
                int filledWidth = (int)(width * getPercentComplete());
                if (filledWidth > 0) {
                    g2d.setColor(getForeground());
                    g2d.fillRoundRect(0, 0, filledWidth, height, cornerRadius, cornerRadius);
                }
            } else { // VERTICAL
                int filledHeight = (int)(height * getPercentComplete());
                if (filledHeight > 0) {
                    // For vertical progress bar, fill from bottom to top
                    int y = height - filledHeight;
                    g2d.setColor(getForeground());
                    g2d.fillRoundRect(0, y, width, filledHeight, cornerRadius, cornerRadius);

                    // Add value indicators (tick marks)
                    g2d.setColor(TEXT_COLOR);
                    for (int i = 0; i <= 4; i++) {
                        int tickY = height - (height * i / 4);
                        g2d.fillRect(width/2 - 15, tickY, 30, 1);
                    }
                }
            }

            // Draw value text
            String valueText = getValue() + "%";
            FontMetrics fm = g2d.getFontMetrics();
            int textWidth = fm.stringWidth(valueText);
            int textHeight = fm.getHeight();

            g2d.setColor(TEXT_COLOR);
            g2d.setFont(new Font("Segoe UI", Font.BOLD, 12));

            if (getOrientation() == SwingConstants.VERTICAL) {
                g2d.drawString(valueText, (width - textWidth) / 2, height - 5);
            } else {
                g2d.drawString(valueText, (width - textWidth) / 2, (height + textHeight) / 2 - 2);
            }

            g2d.dispose();
        }
    }

    public static void main(String[] args) {
        try {
            // Set system look and feel
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

            // Apply nimbus look and feel if available
            for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }

            // Set global UI properties
            UIManager.put("ProgressBar.background", new Color(30, 30, 40));
            UIManager.put("ProgressBar.foreground", new Color(80, 250, 123));
            UIManager.put("ProgressBar.selectionBackground", new Color(80, 250, 123));
            UIManager.put("ProgressBar.selectionForeground", new Color(30, 30, 40));

        } catch (Exception e) {
            e.printStackTrace();
        }

        SwingUtilities.invokeLater(() -> {
            new GestureControlClient();
        });
    }
}