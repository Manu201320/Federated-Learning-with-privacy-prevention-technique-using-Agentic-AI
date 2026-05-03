// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditTrail {

    struct Round {
        uint256 roundNumber;
        string[] banksSelected;
        string modelHash;
        bool anomalyDetected;
        string anomalyBank;
        uint256 timestamp;
    }

    struct TrustScore {
        string bankName;
        uint256 score;
        uint256 timestamp;
    }

    Round[] public rounds;
    mapping(string => TrustScore[]) public trustHistory;
    address public admin;

    event RoundLogged(uint256 roundNumber, string modelHash, uint256 timestamp);
    event AnomalyDetected(uint256 roundNumber, string bankName, uint256 timestamp);
    event TrustScoreUpdated(string bankName, uint256 newScore, uint256 timestamp);

    constructor() {
        admin = msg.sender;
    }

    // 🔐 ACCESS CONTROL
    modifier onlyAdmin() {
        require(msg.sender == admin, "Not authorized");
        _;
    }

    function logRound(
        uint256 roundNumber,
        string[] memory banksSelected,
        string memory modelHash,
        bool anomalyDetected,
        string memory anomalyBank
    ) public onlyAdmin {

        rounds.push(Round({
            roundNumber:     roundNumber,
            banksSelected:   banksSelected,
            modelHash:       modelHash,
            anomalyDetected: anomalyDetected,
            anomalyBank:     anomalyBank,
            timestamp:       block.timestamp
        }));

        emit RoundLogged(roundNumber, modelHash, block.timestamp);

        if (anomalyDetected) {
            emit AnomalyDetected(roundNumber, anomalyBank, block.timestamp);
        }
    }

    function updateTrustScore(string memory bankName, uint256 score)
        public onlyAdmin
    {
        trustHistory[bankName].push(TrustScore({
            bankName:  bankName,
            score:     score,
            timestamp: block.timestamp
        }));

        emit TrustScoreUpdated(bankName, score, block.timestamp);
    }

    function getRoundsCount() public view returns (uint256) {
        return rounds.length;
    }

    function getRound(uint256 index) public view returns (
        uint256, string memory, bool, string memory, uint256
    ) {
        Round memory r = rounds[index];
        return (
            r.roundNumber,
            r.modelHash,
            r.anomalyDetected,
            r.anomalyBank,
            r.timestamp
        );
    }
}