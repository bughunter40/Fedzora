import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';

// AI Avatar Class
class AIAvatar {
    constructor(position) {
        this.entity = document.createElement('a-entity');
        this.entity.setAttribute('geometry', 'primitive: sphere; radius: 0.3');
        this.entity.setAttribute('material', 'color: #00ffff; metalness: 0.8; roughness: 0.2');
        this.entity.setAttribute('position', position);
        this.entity.setAttribute('light', 'type: point; intensity: 0.5; distance: 2; color: #00ffff');
        
        document.querySelector('#patrol-path').appendChild(this.entity);
        this.startPatrol();
    }

    startPatrol() {
        const patrolPoints = this.generatePatrolPath();
        this.animate(patrolPoints);
    }

    generatePatrolPath() {
        const points = [];
        const radius = 4;
        for (let i = 0; i <= 360; i += 45) {
            const angle = (i * Math.PI) / 180;
            points.push(new THREE.Vector3(
                radius * Math.cos(angle),
                0.5,
                radius * Math.sin(angle)
            ));
        }
        return points;
    }

    animate(points) {
        let currentPoint = 0;
        const moveToNextPoint = () => {
            const current = points[currentPoint];
            const next = points[(currentPoint + 1) % points.length];
            
            new TWEEN.Tween(current)
                .to(next, 2000)
                .onUpdate((coords) => {
                    this.entity.setAttribute('position', coords);
                })
                .onComplete(() => {
                    currentPoint = (currentPoint + 1) % points.length;
                    moveToNextPoint();
                })
                .start();
        };
        moveToNextPoint();
    }
}

// Network Visualization
class NetworkVisualizer {
    constructor() {
        this.nodes = [];
        this.createNodes();
        this.simulateTraffic();
    }

    createNodes() {
        const networkContainer = document.createElement('a-entity');
        networkContainer.setAttribute('position', '0 3 -4.9');
        document.querySelector('#soc-room').appendChild(networkContainer);

        for (let i = 0; i < 20; i++) {
            const node = document.createElement('a-sphere');
            node.setAttribute('radius', '0.05');
            node.setAttribute('color', '#00ff00');
            node.setAttribute('position', {
                x: (Math.random() - 0.5) * 7,
                y: (Math.random() - 0.5) * 3,
                z: 0
            });
            networkContainer.appendChild(node);
            this.nodes.push(node);
        }
    }

    simulateTraffic() {
        setInterval(() => {
            this.nodes.forEach(node => {
                if (Math.random() > 0.7) {
                    node.setAttribute('color', '#ffff00');
                    setTimeout(() => node.setAttribute('color', '#00ff00'), 500);
                }
            });
        }, 1000);
    }
}

// Threat Detection System
class ThreatDetection {
    constructor() {
        this.alertElement = this.createAlertElement();
        this.simulateThreats();
    }

    createAlertElement() {
        const alert = document.createElement('div');
        alert.className = 'threat-alert';
        document.body.appendChild(alert);
        return alert;
    }

    simulateThreats() {
        setInterval(() => {
            if (Math.random() > 0.9) {
                this.triggerAlert('Potential security breach detected!');
            }
        }, 10000);
    }

    triggerAlert(message) {
        this.alertElement.textContent = message;
        this.alertElement.classList.add('active');
        setTimeout(() => {
            this.alertElement.classList.remove('active');
        }, 5000);
    }
}

// AI Assistant Interface
class AIAssistant {
    constructor() {
        this.messageContainer = document.querySelector('.ai-message');
        this.messages = [
            'Monitoring network traffic...',
            'Analyzing security patterns...',
            'Scanning for vulnerabilities...',
            'Updating threat database...',
        ];
        this.updateMessages();
    }

    updateMessages() {
        setInterval(() => {
            const randomMessage = this.messages[Math.floor(Math.random() * this.messages.length)];
            this.messageContainer.textContent = randomMessage;
        }, 5000);
    }
}

// Initialize components when A-Frame scene is loaded
document.querySelector('a-scene').addEventListener('loaded', () => {
    // Create AI avatars
    new AIAvatar('0 0.5 0');
    
    // Initialize network visualization
    new NetworkVisualizer();
    
    // Start threat detection system
    new ThreatDetection();
    
    // Initialize AI assistant
    new AIAssistant();
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        TWEEN.update();
    }
    animate();
});