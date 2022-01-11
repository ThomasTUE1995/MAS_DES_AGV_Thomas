from pade.misc.utility import display_message, start_loop, start_single_agent
from pade.core.agent import Agent
from pade.core.new_ams import AMS
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import FipaContractNetProtocol, FipaSubscribeProtocol
from pade.behaviours.protocols import TimedBehaviour

from sys import argv
from random import uniform, random


class SubscriberProtocol(FipaSubscribeProtocol):

    def __init__(self, agent, message):
        super(SubscriberProtocol, self).__init__(agent,
                                                 message,
                                                 is_initiator=True)

    def handle_agree(self, message):
        display_message(self.agent.aid.name, message.content)

    def handle_inform(self, message):
        display_message(self.agent.aid.name, message.content)


class PublisherProtocol(FipaSubscribeProtocol):

    def __init__(self, agent):
        super(PublisherProtocol, self).__init__(agent,
                                                message=None,
                                                is_initiator=False)

    def handle_subscribe(self, message):
        self.register(message.sender)
        display_message(self.agent.aid.name, message.content)
        resposta = message.create_reply()
        resposta.set_performative(ACLMessage.AGREE)
        resposta.set_content('Subscribe message accepted')
        self.agent.send(resposta)

    def handle_cancel(self, message):
        self.deregister(self)
        display_message(self.agent.aid.name, message.content)

    def notify(self, message):
        super(PublisherProtocol, self).notify(message)


class CompContNet1(FipaContractNetProtocol):
    '''CompContNet1

       Initial FIPA-ContractNet Behaviour that sends CFP messages
       to other feeder agents asking for restoration proposals.
       This behaviour also analyzes the proposals and selects the
       one it judges to be the best.'''

    def __init__(self, agent, participants):
        super(CompContNet1, self).__init__(
            agent=agent, message=None, is_initiator=True)
        self.particapants = participants
        # self.message = message

    def handle_all_proposes(self, proposes):
        """
        """

        super(CompContNet1, self).handle_all_proposes(proposes)

        best_proposer = None
        higher_power = 0.0
        other_proposers = list()
        display_message(self.agent.aid.name, 'Analyzing proposals...')

        i = 1

        # logic to select proposals by the higher available power.
        for message in proposes:
            content = message.content
            power = float(content)
            display_message(self.agent.aid.name,
                            'Analyzing proposal {i}'.format(i=i))
            display_message(self.agent.aid.name,
                            'Power Offered: {pot}'.format(pot=power))
            i += 1
            if power > higher_power:
                if best_proposer is not None:
                    other_proposers.append(best_proposer)

                higher_power = power
                best_proposer = message.sender
            else:
                other_proposers.append(message.sender)

        display_message(self.agent.aid.name,
                        'The best proposal was: {pot} VA'.format(
                            pot=higher_power))

        if other_proposers:
            display_message(self.agent.aid.name,
                            'Sending REJECT_PROPOSAL answers...')
            answer = ACLMessage(ACLMessage.REJECT_PROPOSAL)
            answer.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
            answer.set_content('')
            for agent in other_proposers:
                answer.add_receiver(agent)

            self.agent.send(answer)

        if best_proposer is not None:
            display_message(self.agent.aid.name,
                            'Sending ACCEPT_PROPOSAL answer...')

            answer = ACLMessage(ACLMessage.ACCEPT_PROPOSAL)
            answer.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
            answer.set_content('OK')
            answer.add_receiver(best_proposer)
            self.agent.send(answer)

    def handle_inform(self, message):
        """
        """
        super(CompContNet1, self).handle_inform(message)

        display_message(self.agent.aid.name, 'INFORM message received')

    def handle_refuse(self, message):
        """
        """
        super(CompContNet1, self).handle_refuse(message)

        display_message(self.agent.aid.name, 'REFUSE message received')

    def handle_propose(self, message):
        """
        """
        super(CompContNet1, self).handle_propose(message)

        display_message(self.agent.aid.name, 'PROPOSE message received')

    def notify(self, message):
        for sub in self.particapants:
            message.add_receiver(sub)
        super(CompContNet1, self).notify(message)


class CompContNet2(FipaContractNetProtocol):
    '''CompContNet2

       FIPA-ContractNet Participant Behaviour that runs when an agent
       receives a CFP message. A proposal is sent and if it is selected,
       the restrictions are analized to enable the restoration.'''

    def __init__(self, agent):
        super(CompContNet2, self).__init__(agent=agent,
                                           message=None,
                                           is_initiator=False)

    def handle_cfp(self, message):
        """
        """
        self.agent.call_later(1.0, self._handle_cfp, message)

    def _handle_cfp(self, message):
        """
        """
        super(CompContNet2, self).handle_cfp(message)
        self.message = message

        display_message(self.agent.aid.name, 'CFP message received')

        answer = self.message.create_reply()
        answer.set_performative(ACLMessage.PROPOSE)
        pot_disp = uniform(100, 500)
        answer.set_content(str(pot_disp))
        self.agent.send(answer)

    def handle_reject_propose(self, message):
        """
        """
        super(CompContNet2, self).handle_reject_propose(message)

        display_message(self.agent.aid.name,
                        'REJECT_PROPOSAL message received')

    def handle_accept_propose(self, message):
        """
        """
        super(CompContNet2, self).handle_accept_propose(message)

        display_message(self.agent.aid.name,
                        'ACCEPT_PROPOSE message received')

        answer = message.create_reply()
        answer.set_performative(ACLMessage.INFORM)
        answer.set_content('OK')
        self.agent.send(answer)


class TimeSubscribe(TimedBehaviour):

    def __init__(self, agent, notify):
        super(TimeSubscribe, self).__init__(agent, 5)
        self.notify = notify
        self.inc = 0

    def on_time(self):
        super(TimeSubscribe, self).on_time()
        message = ACLMessage(ACLMessage.INFORM)
        message.set_protocol(ACLMessage.FIPA_SUBSCRIBE_PROTOCOL)
        message.set_content(str(random()))
        self.notify(message)
        self.inc += 0.1


class TimedProposals(TimedBehaviour):

    def __init__(self, agent, notify):
        super(TimedProposals, self).__init__(agent, 8.0)
        self.notify = notify
        self.inc = 0

    def on_time(self):
        super(TimedProposals, self).on_time()
        message = ACLMessage(ACLMessage.CFP)
        message.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
        message.set_content('60.0')
        self.notify(message)
        self.inc += 8.0


class TimedCreation(TimedBehaviour):
    """Timed protocal for creating new JAs within the system"""

    def __init__(self, agent):
        super(TimedCreation, self).__init__(agent, 10.0)
        self.inc = 10000

    def on_time(self):
        super(TimedCreation, self).on_time()
        participants1 = list()
        agents_new = list()
        agent_name = 'agent_hello_{}@localhost:{}'.format(self.inc, self.inc)
        participants1.append(agent_name)

        msg = ACLMessage(ACLMessage.SUBSCRIBE)
        msg.set_protocol(ACLMessage.FIPA_SUBSCRIBE_PROTOCOL)
        msg.set_content('Subscription request')
        msg.add_receiver(AID('agent_initiator_20000@localhost:20000'))

        agente_part_1 = AgentHelloWorld(AID(name=agent_name), msg)
        agents_new.append(agente_part_1)
        start_single_agent(agente_part_1)
        self.inc += 100


class AgentHelloWorld(Agent):
    def __init__(self, aid, message):
        super(AgentHelloWorld, self).__init__(aid=aid, debug=False)
        display_message(self.aid.localname, 'Hello World!')
        self.call_later(8.0, self.launch_subscriber_protocol, message)

        # self.protocol = SubscriberProtocol(self, message)
        # self.behaviours.append(self.protocol)
        # self.protocol.on_start()

        # self.pot_disp = uniform(100.0, 500.0)  # Value to send

        # comp = CompContNet2(self)  # Machine Agents Behaviour
        #
        # self.behaviours.append(comp)

    def launch_subscriber_protocol(self, message):
        print(self.agentInstance.table)
        self.protocol = SubscriberProtocol(self, message)
        self.behaviours.append(self.protocol)
        self.protocol.on_start()


class JobPoolAgent(Agent):

    def __init__(self, aid, participants):
        super(JobPoolAgent, self).__init__(aid=aid, debug=False)

        # self.protocol = CompContNet1(self, participants)
        # self.timedCFP = TimedProposals(self, self.protocol.notify)
        self.subProtocal = PublisherProtocol(self)
        self.timed = TimeSubscribe(self, self.subProtocal.notify)

        # self.behaviours.append(self.protocol)
        # self.behaviours.append(self.timedCFP)
        self.behaviours.append(self.subProtocal)
        self.behaviours.append(self.timed)


class MachineAgent(Agent):

    def __init__(self, aid, pot_disp):
        super(MachineAgent, self).__init__(aid=aid, debug=False)

        self.pot_disp = uniform(100.0, 500.0)  # Value to send

        comp = CompContNet2(self)  # Machine Agents Behaviour

        self.behaviours.append(comp)


class AgentCreator(Agent):

    def __init__(self, aid):
        super(AgentCreator, self).__init__(aid=aid, debug=False)
        self.behaviours.append(TimedCreation(self))


if __name__ == "__main__":
    agents_per_process = 1
    c = 0
    agents = list()
    for i in range(agents_per_process):
        port = int(argv[1]) + c
        k = 1000
        participants = list()

        # agent_name = 'agent_participant_{}@localhost:{}'.format(port - k, port - k)
        # participants.append(agent_name)
        # agente_part_1 = MachineAgent(AID(name=agent_name), uniform(100.0, 500.0))
        # agents.append(agente_part_1)
        #
        # agent_name = 'agent_participant_{}@localhost:{}'.format(port + k, port + k)
        # participants.append(agent_name)
        # agente_part_2 = MachineAgent(AID(name=agent_name), uniform(100.0, 500.0))
        # agents.append(agente_part_2)

        agent_name = 'agent_initiator_{}@localhost:{}'.format(port, port)
        agente_init_1 = JobPoolAgent(AID(name=agent_name), participants)
        agents.append(agente_init_1)

        agent_name = 'agent_creator_{}@localhost:{}'.format(port + 2, port + 2)
        agente_init_1 = AgentCreator(AID(name=agent_name))
        agents.append(agente_init_1)

        c += 1000

    start_loop(agents)
