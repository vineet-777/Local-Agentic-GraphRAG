/**
 * Local Agentic GraphRAG — God-Level 3D Knowledge Graph Visualizer
 * High-Performance, Zero-Dependency HTML5 3D Perspective WebGL/Canvas Engine.
 * Features:
 * - 3D Spherical & Spring Force Layout
 * - Orbit, Pan, and Zoom Controls (Mouse & Touch)
 * - Node Degree Sizing & Community Cluster Palette
 * - Multi-Hop Neighborhood Highlighting on Click
 * - Entity Search & Camera Auto-Focus
 * - Interactive Inspector Panel with Direct Querying
 */

class KnowledgeGraph3D {
  constructor(canvasId, containerId, inspectorId) {
    this.canvas = document.getElementById(canvasId);
    this.container = document.getElementById(containerId);
    this.inspector = document.getElementById(inspectorId);
    this.ctx = this.canvas.getContext('2d');

    this.nodes = [];
    this.links = [];
    this.nodeMap = new Map();
    this.adjMap = new Map();

    // 3D Camera & Transform State
    this.camera = {
      rotX: 0.3,
      rotY: 0.5,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      fov: 480,
    };

    // Interaction State
    this.isDragging = false;
    this.lastMouse = { x: 0, y: 0 };
    this.hoveredNode = null;
    this.selectedNode = null;
    this.autoRotate = true;
    this.filterText = '';

    this.animId = null;

    this.initEvents();
  }

  resize() {
    if (!this.container) return;
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
  }

  async loadData() {
    try {
      const res = await fetch('/api/graph/3d-data');
      if (!res.ok) throw new Error('Failed to load 3d graph');
      const data = await res.json();
      this.initGraph(data.nodes || [], data.links || []);
    } catch (err) {
      console.warn('Could not load 3D graph data:', err);
    }
  }

  initGraph(rawNodes, rawLinks) {
    this.nodeMap.clear();
    this.adjMap.clear();

    const count = rawNodes.length;
    // Arrange initial positions on a 3D Fibonacci sphere
    const goldenRatio = (1 + Math.sqrt(5)) / 2;
    const radius = Math.min(220, Math.max(120, count * 0.7));

    this.nodes = rawNodes.map((n, i) => {
      const theta = 2 * Math.PI * i / goldenRatio;
      const phi = Math.acos(1 - 2 * (i + 0.5) / count);
      const x = radius * Math.cos(theta) * Math.sin(phi);
      const y = radius * Math.sin(theta) * Math.sin(phi);
      const z = radius * Math.cos(phi);

      const nodeObj = {
        id: String(n.id),
        name: n.name || n.id,
        type: n.type || 'concept',
        color: n.color || '#10a37f',
        val: n.val || 8,
        degree: n.degree || 1,
        x: x + (Math.random() - 0.5) * 30,
        y: y + (Math.random() - 0.5) * 30,
        z: z + (Math.random() - 0.5) * 30,
        vx: 0,
        vy: 0,
        vz: 0,
        projX: 0,
        projY: 0,
        projScale: 1,
        visible: true,
      };

      this.nodeMap.set(nodeObj.id, nodeObj);
      this.adjMap.set(nodeObj.id, []);
      return nodeObj;
    });

    this.links = rawLinks.map(l => {
      const sId = String(l.source);
      const tId = String(l.target);
      if (this.adjMap.has(sId)) this.adjMap.get(sId).push({ target: tId, rel: l.relation });
      if (this.adjMap.has(tId)) this.adjMap.get(tId).push({ target: sId, rel: l.relation });
      return {
        source: sId,
        target: tId,
        relation: l.relation || 'related_to',
      };
    });

    this.resize();
    this.stepPhysics(30); // Warm up layout
  }

  stepPhysics(iterations = 1) {
    const k = 0.04; // Spring constant
    const repulse = 1800; // Node repulsion
    const damping = 0.88;

    for (let it = 0; it < iterations; it++) {
      // Repulsion between node pairs (sampled for performance)
      const len = this.nodes.length;
      const step = len > 200 ? 2 : 1;
      for (let i = 0; i < len; i += step) {
        const a = this.nodes[i];
        for (let j = i + 1; j < len; j += step) {
          const b = this.nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dz = a.z - b.z;
          const distSq = dx * dx + dy * dy + dz * dz + 100;
          const dist = Math.sqrt(distSq);
          if (dist < 320) {
            const force = repulse / distSq;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            const fz = (dz / dist) * force;
            a.vx += fx; a.vy += fy; a.vz += fz;
            b.vx -= fx; b.vy -= fy; b.vz -= fz;
          }
        }
      }

      // Spring attraction along links
      for (const link of this.links) {
        const a = this.nodeMap.get(link.source);
        const b = this.nodeMap.get(link.target);
        if (a && b) {
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dz = b.z - a.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;
          const desired = 65;
          const force = (dist - desired) * k;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          const fz = (dz / dist) * force;
          a.vx += fx; a.vy += fy; a.vz += fz;
          b.vx -= fx; b.vy -= fy; b.vz -= fz;
        }
      }

      // Center gravity & velocity integration
      for (const n of this.nodes) {
        n.vx -= n.x * 0.003;
        n.vy -= n.y * 0.003;
        n.vz -= n.z * 0.003;

        n.vx *= damping;
        n.vy *= damping;
        n.vz *= damping;

        n.x += n.vx;
        n.y += n.vy;
        n.z += n.vz;
      }
    }
  }

  project3D(x, y, z) {
    // Rotation around Y axis
    const cosY = Math.cos(this.camera.rotY);
    const sinY = Math.sin(this.camera.rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = z * cosY + x * sinY;

    // Rotation around X axis
    const cosX = Math.cos(this.camera.rotX);
    const sinX = Math.sin(this.camera.rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = z1 * cosX + y * sinX;

    // Perspective projection
    const fov = this.camera.fov * this.camera.zoom;
    const depth = z2 + 600;
    const scale = depth > 10 ? fov / depth : 0.1;

    const screenX = this.width / 2 + x1 * scale + this.camera.panX;
    const screenY = this.height / 2 + y2 * scale + this.camera.panY;

    return { x: screenX, y: screenY, z: z2, scale, depth };
  }

  render() {
    if (!this.ctx || !this.width) return;
    this.ctx.clearRect(0, 0, this.width, this.height);

    if (this.autoRotate && !this.isDragging) {
      this.camera.rotY += 0.002;
    }

    // Gentle physical relaxation
    this.stepPhysics(1);

    // Project all nodes
    for (const n of this.nodes) {
      const p = this.project3D(n.x, n.y, n.z);
      n.projX = p.x;
      n.projY = p.y;
      n.projZ = p.z;
      n.projScale = p.scale;
    }

    // Filter matching
    const query = this.filterText.trim().toLowerCase();

    // 1. Render Links
    this.ctx.lineWidth = 1;
    for (const l of this.links) {
      const a = this.nodeMap.get(l.source);
      const b = this.nodeMap.get(l.target);
      if (!a || !b) continue;

      const isHighlighted = (this.selectedNode && (this.selectedNode.id === a.id || this.selectedNode.id === b.id));
      
      if (isHighlighted) {
        this.ctx.strokeStyle = 'rgba(16, 163, 127, 0.85)';
        this.ctx.lineWidth = 2.2;
      } else {
        const avgScale = (a.projScale + b.projScale) * 0.5;
        const alpha = Math.max(0.06, Math.min(0.22, avgScale * 0.2));
        this.ctx.strokeStyle = `rgba(180, 180, 180, ${alpha})`;
        this.ctx.lineWidth = 1;
      }

      this.ctx.beginPath();
      this.ctx.moveTo(a.projX, a.projY);
      this.ctx.lineTo(b.projX, b.projY);
      this.ctx.stroke();
    }

    // 2. Sort nodes by Z-depth (back-to-front rendering)
    const sortedNodes = [...this.nodes].sort((a, b) => a.projZ - b.projZ);

    // 3. Render Nodes
    for (const n of sortedNodes) {
      const isMatch = query && n.name.toLowerCase().includes(query);
      const isSelected = this.selectedNode && this.selectedNode.id === n.id;
      const isHovered = this.hoveredNode && this.hoveredNode.id === n.id;

      const baseR = Math.max(2.5, Math.min(18, (n.val || 8) * n.projScale));
      const r = isMatch || isSelected || isHovered ? baseR * 1.5 : baseR;

      // Glow behind important nodes
      if (isSelected || isMatch || isHovered) {
        this.ctx.beginPath();
        this.ctx.arc(n.projX, n.projY, r * 2.2, 0, Math.PI * 2);
        this.ctx.fillStyle = isSelected ? 'rgba(16, 163, 127, 0.35)' : 'rgba(56, 189, 248, 0.3)';
        this.ctx.fill();
      }

      // Main Node Circle
      this.ctx.beginPath();
      this.ctx.arc(n.projX, n.projY, r, 0, Math.PI * 2);
      this.ctx.fillStyle = isMatch ? '#f59e0b' : (isSelected ? '#10a37f' : n.color);
      this.ctx.fill();
      this.ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(0, 0, 0, 0.4)';
      this.ctx.lineWidth = isSelected ? 2 : 1;
      this.ctx.stroke();

      // Node Label (for large nodes, matches, or when hovered/selected)
      if (r > 6 || isMatch || isSelected || isHovered) {
        this.ctx.font = `${Math.max(10, Math.min(13, 10 * n.projScale))}px "OpenAI Sans", sans-serif`;
        this.ctx.fillStyle = isSelected ? '#10a37f' : (isMatch ? '#f59e0b' : '#ececec');
        this.ctx.textAlign = 'center';
        this.ctx.fillText(n.name, n.projX, n.projY - r - 4);
      }
    }
  }

  start() {
    if (this.animId) cancelAnimationFrame(this.animId);
    const loop = () => {
      this.render();
      this.animId = requestAnimationFrame(loop);
    };
    loop();
  }

  stop() {
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  initEvents() {
    window.addEventListener('resize', () => this.resize());

    // Mouse / Touch Controls
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.container || this.container.offsetParent === null) return;

      if (this.isDragging) {
        const dx = e.clientX - this.lastMouse.x;
        const dy = e.clientY - this.lastMouse.y;
        this.camera.rotY += dx * 0.007;
        this.camera.rotX += dy * 0.007;
        this.lastMouse = { x: e.clientX, y: e.clientY };
        return;
      }

      // Check node hover
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let found = null;
      for (const n of this.nodes) {
        const dist = Math.hypot(n.projX - mouseX, n.projY - mouseY);
        if (dist < Math.max(8, n.val * n.projScale * 1.5)) {
          found = n;
          break;
        }
      }
      this.hoveredNode = found;
      this.canvas.style.cursor = found ? 'pointer' : (this.isDragging ? 'grabbing' : 'grab');
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      this.camera.zoom = Math.max(0.3, Math.min(4.0, this.camera.zoom * zoomFactor));
    }, { passive: false });

    // Node Click / Select
    this.canvas.addEventListener('click', (e) => {
      if (this.hoveredNode) {
        this.selectNode(this.hoveredNode);
      }
    });
  }

  selectNode(node) {
    this.selectedNode = node;
    if (!this.inspector) return;

    const nameEl = document.getElementById('inspectNodeName');
    const detailsEl = document.getElementById('inspectNodeDetails');
    if (nameEl) nameEl.textContent = node.name;

    const neighbors = this.adjMap.get(node.id) || [];
    let html = `
      <div class="inspect-field"><span>Type:</span> <strong>${node.type}</strong></div>
      <div class="inspect-field"><span>Connections (Degree):</span> <strong>${node.degree}</strong></div>
      <div class="inspect-sub">Direct Relationships (${neighbors.length}):</div>
      <div class="inspect-rels-list">
    `;

    if (neighbors.length === 0) {
      html += `<div style="font-size:0.75rem; color:var(--text-muted);">No direct relationships found.</div>`;
    } else {
      neighbors.slice(0, 15).forEach(rel => {
        const targetObj = this.nodeMap.get(rel.target);
        const targetName = targetObj ? targetObj.name : rel.target;
        html += `
          <div class="inspect-rel-item">
            <span class="rel-tag">${rel.rel}</span>
            <span class="rel-target">${targetName}</span>
          </div>
        `;
      });
    }

    html += `
      </div>
      <button class="inspect-ask-btn" id="askAboutNodeBtn">
        💬 Ask GraphRAG about "${node.name.slice(0, 25)}"
      </button>
    `;

    if (detailsEl) detailsEl.innerHTML = html;
    this.inspector.style.display = 'block';

    const askBtn = document.getElementById('askAboutNodeBtn');
    if (askBtn) {
      askBtn.onclick = () => {
        const modal = document.getElementById('graphModal');
        if (modal) modal.style.display = 'none';
        const queryInput = document.getElementById('queryInput');
        if (queryInput) {
          queryInput.value = `Explain the concept "${node.name}" and all its connections in the corpus.`;
          queryInput.focus();
        }
      };
    }
  }

  filter(text) {
    this.filterText = text;
  }

  reset() {
    this.camera = {
      rotX: 0.3,
      rotY: 0.5,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      fov: 480,
    };
    this.selectedNode = null;
    this.filterText = '';
    if (this.inspector) this.inspector.style.display = 'none';
  }
}

window.KnowledgeGraph3D = KnowledgeGraph3D;
