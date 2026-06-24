TODO:
- [ ] Make RayNet training environments protocol-agnostic. The omnetpp component used should be determined by the selected state, and added as a replacement by the RayNet env.py before spawning an simulation.
- [ ] Implement synchronized stepping (blocking) for simulations to prevent quicker environments from saturating the replay buffer.
- [ ] Find a way to implement blocking in simulations while maintaining high CPU utilization
- [ ] Add an entry point for inference-only runs