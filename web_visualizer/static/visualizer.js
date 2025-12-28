// Catan board visualizer with Socket.IO
const socket = io();

const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');

// Canvas settings
const HEX_RADIUS = 60;
const HEX_HEIGHT = Math.sqrt(3) * HEX_RADIUS;
const HEX_WIDTH = 2 * HEX_RADIUS;

let gameState = null;
let actionLog = [];

// Color mappings
const RESOURCE_COLORS = {
    'WOOD': '#8d6e63',
    'BRICK': '#d84315',
    'SHEEP': '#81c784',
    'WHEAT': '#ffd54f',
    'ORE': '#616161',
    'DESERT': '#f4e4c1'
};

const PLAYER_COLORS = {
    'BLUE': '#2196F3',
    'RED': '#f44336',
    'ORANGE': '#ff9800',
    'WHITE': '#9e9e9e'
};

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
});

socket.on('game_state', (state) => {
    console.log('Received game state:', state);
    gameState = state;
    updateUI();
    drawGame();
});

socket.on('game_end', (result) => {
    console.log('Game ended:', result);
    // Winner banner removed - game continues to next episode
});

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-status');
    const statusText = document.getElementById('status-text');
    
    if (connected) {
        indicator.classList.remove('disconnected');
        indicator.classList.add('connected');
        statusText.textContent = 'Connected - Waiting for game...';
    } else {
        indicator.classList.remove('connected');
        indicator.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}

function updateUI() {
    if (!gameState) return;

    // Update game info
    document.getElementById('episode').textContent = gameState.episode || 0;
    document.getElementById('step').textContent = gameState.step || 0;
    document.getElementById('current-player').textContent = gameState.current_player || '-';
    
    // Update dice roll
    const diceElement = document.getElementById('dice-roll');
    if (gameState.dice_roll && gameState.dice_roll.length === 2) {
        const [die1, die2] = gameState.dice_roll;
        const total = die1 + die2;
        diceElement.textContent = `🎲 ${die1} + ${die2} = ${total}`;
        diceElement.style.color = (total === 7) ? '#d32f2f' : '#333';
    } else {
        diceElement.textContent = '🎲 - -';
        diceElement.style.color = '#666';
    }

    // Update players
    updatePlayers();

    // Update action log
    if (gameState.action) {
        addActionLog(gameState.action);
    }

    // Winner logged in console instead of banner
    if (gameState.game_over && gameState.winner) {
        console.log('Game Over! Winner:', gameState.winner);
    }
}

function updatePlayers() {
    const container = document.getElementById('players-container');
    container.innerHTML = '';

    if (!gameState || !gameState.players) return;

    gameState.players.forEach(player => {
        const card = document.createElement('div');
        card.className = `player-card ${player.color}`;
        
        const resources = Object.entries(player.resources)
            .filter(([_, count]) => count > 0)
            .map(([res, count]) => `<span class="resource-chip ${res}">${res}: ${count}</span>`)
            .join('');

        card.innerHTML = `
            <div class="player-name">${player.name}</div>
            <div class="player-stats">
                <p>🏆 Victory Points: ${player.victory_points}</p>
                <p>🃏 Dev Cards: ${player.dev_cards}</p>
                <p>🛤️ Roads: ${player.roads_left} | 🏠 Settlements: ${player.settlements_left} | 🏰 Cities: ${player.cities_left}</p>
                ${player.longest_road ? '<p>🛤️ Longest Road</p>' : ''}
                ${player.largest_army ? '<p>⚔️ Largest Army</p>' : ''}
            </div>
            <div class="resources">${resources}</div>
        `;
        
        container.appendChild(card);
    });
}

function addActionLog(action) {
    actionLog.unshift(action);
    if (actionLog.length > 10) actionLog.pop();

    const logContainer = document.getElementById('action-log-entries');
    logContainer.innerHTML = actionLog.map(log => 
        `<div class="log-entry">${log}</div>`
    ).join('');
}

function showWinnerBanner(winner) {
    const banner = document.getElementById('winner-banner');
    const winnerName = document.getElementById('winner-name');
    winnerName.textContent = winner;
    banner.classList.add('show');
}

// Drawing functions
function resizeCanvas() {
    const container = document.getElementById('canvas-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    if (gameState) drawGame();
}

function axialToPixel(q, r) {
    // Convert cube/axial coordinates to pixel coordinates
    // Catanatron Coordinate System (Pointy Top)
    // EAST (1, -1, 0) -> 0 degrees (Right)
    // NORTHEAST (1, 0, -1) -> -60 degrees (Top Right)
    // Derived formula:
    // x = radius * sqrt(3)/2 * (q - r)
    // y = radius * -3/2 * (q + r)
    const x = HEX_RADIUS * (Math.sqrt(3)/2 * (q - r));
    const y = HEX_RADIUS * (-1.5 * (q + r));
    return { x, y };
}

function drawHexagon(x, y, radius, color, strokeColor = '#333', strokeWidth = 2) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        const angle = Math.PI / 3 * i - Math.PI / 6;
        const hx = x + radius * Math.cos(angle);
        const hy = y + radius * Math.sin(angle);
        if (i === 0) {
            ctx.moveTo(hx, hy);
        } else {
            ctx.lineTo(hx, hy);
        }
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.stroke();
}

function drawNumber(x, y, number, hasRobber = false) {
    // Draw number token
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, 2 * Math.PI);
    ctx.fillStyle = hasRobber ? '#333' : '#f5f5f5';
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw number
    ctx.fillStyle = hasRobber ? '#fff' : (number === 6 || number === 8 ? '#d32f2f' : '#333');
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(number || '', x, y);

    // Draw robber if present
    if (hasRobber) {
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px Arial';
        ctx.fillText('🎭', x, y - 25);
    }
}

function drawSettlement(x, y, color) {
    ctx.save();
    ctx.translate(x, y);
    
    // Draw house shape - larger and more visible
    ctx.beginPath();
    ctx.moveTo(0, -15);  // Top point
    ctx.lineTo(12, -6);   // Top right
    ctx.lineTo(12, 12);   // Bottom right
    ctx.lineTo(-12, 12);  // Bottom left
    ctx.lineTo(-12, -6);  // Top left
    ctx.closePath();
    
    ctx.fillStyle = PLAYER_COLORS[color] || color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Add a border for better visibility
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    ctx.restore();
}

function drawCity(x, y, color) {
    ctx.save();
    ctx.translate(x, y);
    
    // Draw larger building - more visible
    ctx.fillStyle = PLAYER_COLORS[color] || color;
    ctx.fillRect(-15, -15, 30, 30);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.strokeRect(-15, -15, 30, 30);
    
    // Draw tower
    ctx.fillRect(-6, -25, 12, 12);
    ctx.strokeRect(-6, -25, 12, 12);
    
    // Black border
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.strokeRect(-15, -15, 30, 30);
    ctx.strokeRect(-6, -25, 12, 12);
    
    ctx.restore();
}

function drawRoad(x1, y1, x2, y2, color) {
    // Draw white outline first
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Draw colored road on top
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = PLAYER_COLORS[color] || color;
    ctx.lineWidth = 7;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Black border for definition
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawGame() {
    if (!gameState || !gameState.tiles) {
        // Draw placeholder
        ctx.fillStyle = '#f5f5f5';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#999';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for game to start...', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Clear canvas
    ctx.fillStyle = '#87CEEB';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Center the board
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.save();
    ctx.translate(centerX, centerY);

    // Draw tiles
    gameState.tiles.forEach(tile => {
        const [q, r, s] = tile.coordinate;
        const pos = axialToPixel(q, r);
        const color = RESOURCE_COLORS[tile.resource] || '#ccc';
        
        drawHexagon(pos.x, pos.y, HEX_RADIUS - 2, color);
        
        if (tile.number) {
            drawNumber(pos.x, pos.y, tile.number, tile.has_robber);
        }

        // Draw resource name
        ctx.fillStyle = '#333';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(tile.resource, pos.x, pos.y + 30);
    });

    // Draw roads
    if (gameState.edges) {
        gameState.edges.forEach(edge => {
            if (edge.coordinates && edge.coordinates.length === 2) {
                const [x1, y1] = edge.coordinates[0];
                const [x2, y2] = edge.coordinates[1];
                drawRoad(x1, y1, x2, y2, edge.color);
            }
        });
    }

    // Draw settlements and cities
    if (gameState.nodes) {
        gameState.nodes.forEach(node => {
            const [x, y] = node.coordinate;
            
            if (node.building_type === 'SETTLEMENT') {
                drawSettlement(x, y, node.color);
            } else if (node.building_type === 'CITY') {
                drawCity(x, y, node.color);
            }
        });
    }

    ctx.restore();
}

// Initialize
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
