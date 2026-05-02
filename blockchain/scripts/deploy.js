async function main() {
    const AuditTrail = await ethers.getContractFactory("AuditTrail");
    const contract = await AuditTrail.deploy();

    await contract.deployed();

    console.log("AuditTrail deployed to:", contract.address);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});